"""FastAPI web server for weight tuner interface."""

import asyncio
import logging
import time
import os
import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np

# Add system dist-packages to path for libcamera access
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

from .edge_detection import EdgeDetector
from .config import CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX, STEERING_ANGLE_CLASSES
from .weight_tuner import SteeringDataCollector, WeightOptimizer, save_weights_to_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weight Tuner Console")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
detector: Optional[EdgeDetector] = None
collector: Optional[SteeringDataCollector] = None
optimizer: Optional[WeightOptimizer] = None
camera_available = False
current_human_angle = 0
optimization_result: Optional[dict] = None
is_recording = False


class SetAngleRequest(BaseModel):
    angle: int

# Mount static files if they exist
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# WebSocket connections
tuner_connections: List[WebSocket] = []


def frame_to_jpeg(frame: np.ndarray) -> bytes:
    """Convert numpy array frame to JPEG bytes."""
    if frame is None:
        return b""
    
    if len(frame.shape) == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
    
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


async def broadcast_tuner_status():
    """Broadcast current tuner status to all connected WebSocket clients."""
    if not collector or not detector:
        return
    
    # Get current frame and calculate prediction
    frame = detector.capture_frame()
    densities = (0.0, 0.0, 0.0, 0.0, 0.0)
    predicted_angle = 0
    
    if frame is not None:
        edges = detector.detect_edges(frame)
        if edges is not None:
            densities = detector.calculate_region_densities(edges)
            predicted_angle, _ = detector.calculate_steering_angle(edges)
    
    status = {
        "type": "status",
        "human_angle": current_human_angle,
        "predicted_angle": predicted_angle,
        "sample_count": collector.get_sample_count(),
        "is_recording": is_recording,
        "densities": {
            "far_left": float(densities[0]),
            "left": float(densities[1]),
            "center": float(densities[2]),
            "right": float(densities[3]),
            "far_right": float(densities[4])
        },
        "optimization_result": optimization_result
    }
    
    disconnected = []
    for connection in tuner_connections:
        try:
            await connection.send_json(status)
        except:
            disconnected.append(connection)
    
    for conn in disconnected:
        if conn in tuner_connections:
            tuner_connections.remove(conn)


@app.get("/")
async def root():
    """Serve the tuner HTML page."""
    # Try tuner.html first, then fallback to index.html
    tuner_path = static_dir / "tuner.html"
    if tuner_path.exists():
        return FileResponse(str(tuner_path))
    
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    
    # Fallback HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Weight Tuner Console</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        <h1>Weight Tuner Console</h1>
        <p>Please build the frontend first: cd frontend && npm install && npm run build</p>
    </body>
    </html>
    """
    return StreamingResponse(iter([html_content]), media_type="text/html")


@app.get("/api/video_feed")
async def video_feed():
    """MJPEG stream of camera feed."""
    if not camera_available or detector is None:
        return StreamingResponse(
            iter([b'--frame\r\nContent-Type: image/jpeg\r\n\r\n']),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    async def generate():
        while True:
            if detector is None or not camera_available:
                await asyncio.sleep(0.1)
                continue
            frame = detector.capture_frame()
            if frame is not None:
                jpeg_bytes = frame_to_jpeg(frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            await asyncio.sleep(1/30)  # ~30 fps
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/edge_feed")
async def edge_feed(edges_only: bool = False):
    """MJPEG stream of edge detection visualization."""
    if not camera_available or detector is None:
        return StreamingResponse(
            iter([b'--frame\r\nContent-Type: image/jpeg\r\n\r\n']),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    async def generate():
        while True:
            if detector is None or not camera_available:
                await asyncio.sleep(0.1)
                continue
            try:
                frame = detector.capture_frame()
                if frame is not None:
                    edges = detector.detect_edges(frame)
                    if edges is not None:
                        visualization = detector.get_edge_visualization(frame, edges, edges_only=edges_only)
                        if visualization is not None:
                            jpeg_bytes = frame_to_jpeg(visualization)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error in edge feed: {e}", exc_info=True)
            await asyncio.sleep(1/30)  # ~30 fps
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/set_angle")
async def set_angle(request: SetAngleRequest):
    """Set the current human steering angle."""
    global current_human_angle
    angle = request.angle
    if angle not in STEERING_ANGLE_CLASSES:
        raise HTTPException(status_code=400, detail=f"Invalid angle. Must be one of {STEERING_ANGLE_CLASSES}")
    
    current_human_angle = angle
    if collector:
        collector.set_human_angle(angle)
    
    await broadcast_tuner_status()
    return {"success": True, "angle": angle}


@app.post("/api/start_recording")
async def start_recording():
    """Start recording samples."""
    global is_recording
    is_recording = True
    await broadcast_tuner_status()
    return {"success": True, "recording": True}


@app.post("/api/stop_recording")
async def stop_recording():
    """Stop recording samples."""
    global is_recording
    is_recording = False
    await broadcast_tuner_status()
    return {"success": True, "recording": False}


@app.get("/api/recording_status")
async def get_recording_status():
    """Get current recording status."""
    return {"recording": is_recording}


@app.post("/api/run_optimization")
async def run_optimization():
    """Run Bayesian optimization."""
    global optimization_result
    
    if not collector or not optimizer:
        raise HTTPException(status_code=500, detail="Collector or optimizer not initialized")
    
    sample_count = collector.get_sample_count()
    if sample_count < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient samples: {sample_count}. Need at least 10 samples."
        )
    
    try:
        logger.info(f"Starting optimization with {sample_count} samples...")
        weights, mae = optimizer.optimize(verbose=False)
        optimization_result = {
            "weights": weights,
            "mae": float(mae),
            "outer": float(weights[0]),
            "inner": float(weights[1]),
            "center": float(weights[2])
        }
        
        await broadcast_tuner_status()
        return {"success": True, "result": optimization_result}
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/api/save_weights")
async def save_weights():
    """Save optimized weights to config."""
    global optimization_result
    
    if not optimization_result:
        raise HTTPException(status_code=400, detail="No optimization result to save. Run optimization first.")
    
    try:
        success = save_weights_to_config(optimization_result["weights"])
        if success:
            return {"success": True, "message": "Weights saved successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save weights to config")
    except Exception as e:
        logger.error(f"Failed to save weights: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save weights: {str(e)}")


@app.post("/api/reset_samples")
async def reset_samples():
    """Reset/clear all collected samples."""
    global optimization_result
    
    if collector:
        collector.clear()
    optimization_result = None
    
    await broadcast_tuner_status()
    return {"success": True, "message": "Samples cleared"}


@app.get("/api/status")
async def get_status():
    """Get current tuner status."""
    if not collector or not detector:
        return {
            "camera_available": camera_available,
            "sample_count": 0,
            "human_angle": 0,
            "predicted_angle": 0,
            "optimization_result": None
        }
    
    frame = detector.capture_frame()
    densities = (0.0, 0.0, 0.0, 0.0, 0.0)
    predicted_angle = 0
    
    if frame is not None:
        edges = detector.detect_edges(frame)
        if edges is not None:
            densities = detector.calculate_region_densities(edges)
            predicted_angle, _ = detector.calculate_steering_angle(edges)
    
    return {
        "camera_available": camera_available,
        "sample_count": collector.get_sample_count(),
        "human_angle": current_human_angle,
        "predicted_angle": predicted_angle,
        "is_recording": is_recording,
        "densities": {
            "far_left": float(densities[0]),
            "left": float(densities[1]),
            "center": float(densities[2]),
            "right": float(densities[3]),
            "far_right": float(densities[4])
        },
        "optimization_result": optimization_result
    }


@app.websocket("/api/tuner")
async def tuner_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time tuner status."""
    await websocket.accept()
    tuner_connections.append(websocket)
    
    try:
        # Send initial status
        await broadcast_tuner_status()
        
        # Keep connection alive and broadcast updates
        while True:
            # Broadcast status every 0.1 seconds
            await broadcast_tuner_status()
            await asyncio.sleep(0.1)
            
            # Also collect samples if camera is available and recording
            if camera_available and detector and collector and is_recording:
                frame = detector.capture_frame()
                if frame is not None:
                    edges = detector.detect_edges(frame)
                    if edges is not None:
                        densities = detector.calculate_region_densities(edges)
                        collector.add_sample(densities)
    except WebSocketDisconnect:
        if websocket in tuner_connections:
            tuner_connections.remove(websocket)
    except Exception as e:
        logger.error(f"Error in tuner WebSocket: {e}", exc_info=True)
        if websocket in tuner_connections:
            tuner_connections.remove(websocket)


@app.on_event("startup")
async def startup_event():
    """Initialize camera and tuner components on startup."""
    global detector, collector, optimizer, camera_available, current_human_angle
    
    try:
        detector = EdgeDetector()
        detector.initialize_camera(CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX)
        camera_available = True
        logger.info("Camera initialized successfully")
    except Exception as e:
        camera_available = False
        logger.error(f"Failed to initialize camera: {e}")
        logger.warning("Server will continue without camera. Some features will be unavailable.")
    
    # Initialize tuner components
    collector = SteeringDataCollector(buffer_size=1000)
    optimizer = WeightOptimizer(collector, n_trials=30)
    current_human_angle = 0
    collector.set_human_angle(current_human_angle)
    logger.info("Tuner components initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global detector
    if detector is not None:
        detector.cleanup()
        logger.info("Server shutting down")

