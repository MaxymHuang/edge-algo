"""FastAPI web server for console interface."""

import asyncio
import logging
import time
import os
import sys
from pathlib import Path
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np

# Add system dist-packages to path for libcamera access
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

from .lane_detector import LaneDetector
from .config import (
    CAMERA_RESOLUTION, 
    CAMERA_FRAMERATE, 
    CAMERA_INDEX,
    LANE_DETECTION_METHOD
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vision Algorithm Console")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: LaneDetector = None
camera_available = False
current_detection_method = LANE_DETECTION_METHOD

# Mount static files if they exist
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    # Mount assets directory for JS/CSS files
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# Log messages storage
log_messages: List[dict] = []
MAX_LOG_MESSAGES = 1000

# WebSocket connections
steering_connections: List[WebSocket] = []
log_connections: List[WebSocket] = []


class WebLogHandler(logging.Handler):
    """Custom logging handler that sends logs to web console."""
    
    def emit(self, record):
        """Emit a log record."""
        try:
            level = record.levelname
            message = self.format(record)
            add_log_message(level, message)
        except Exception:
            self.handleError(record)


# Add web log handler to root logger
web_handler = WebLogHandler()
web_handler.setFormatter(logging.Formatter('%(message)s'))
logging.getLogger().addHandler(web_handler)


def add_log_message(level: str, message: str):
    """Add a log message to the storage."""
    log_entry = {
        "timestamp": time.time(),
        "level": level,
        "message": message
    }
    log_messages.append(log_entry)
    if len(log_messages) > MAX_LOG_MESSAGES:
        log_messages.pop(0)
    
    # Broadcast to all connected log WebSocket clients
    asyncio.create_task(broadcast_log_message(log_entry))


async def broadcast_log_message(log_entry: dict):
    """Broadcast log message to all connected WebSocket clients."""
    disconnected = []
    for connection in log_connections:
        try:
            await connection.send_json(log_entry)
        except:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in log_connections:
            log_connections.remove(conn)


async def broadcast_steering(steering_data: dict):
    """Broadcast steering direction and angle to all connected WebSocket clients.
    
    Args:
        steering_data: Dictionary with "direction" and "angle" keys
    """
    disconnected = []
    for connection in steering_connections:
        try:
            await connection.send_json(steering_data)
        except:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        if conn in steering_connections:
            steering_connections.remove(conn)


def frame_to_jpeg(frame: np.ndarray) -> bytes:
    """Convert numpy array frame to JPEG bytes."""
    if frame is None:
        return b""
    
    # Convert RGB to BGR for OpenCV
    if len(frame.shape) == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
    
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        # Fallback HTML if static files not built
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vision Algorithm Console</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body>
            <h1>Vision Algorithm Console</h1>
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
    """MJPEG stream of edge detection visualization.
    
    Args:
        edges_only: If True, show only edges on black background. If False, overlay edges on original frame.
    """
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
                        else:
                            logger.debug("Edge visualization is None")
                    else:
                        logger.debug("Edge detection returned None")
                else:
                    # Frame capture failed - this can happen if camera is busy or temporarily unavailable
                    # Only log as debug to avoid spam, but check camera status periodically
                    logger.debug("Frame capture returned None (camera may be busy)")
            except Exception as e:
                logger.error(f"Error in edge feed: {e}", exc_info=True)
            await asyncio.sleep(1/30)  # ~30 fps
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/ml_feed")
async def ml_feed(model_type: str = "fast_scnn"):
    """MJPEG stream of ML lane detection visualization.
    
    Args:
        model_type: "fast_scnn" or "yolo"
    """
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
                    visualization = detector.get_ml_visualization(frame, model_type=model_type)
                    if visualization is not None:
                        jpeg_bytes = frame_to_jpeg(visualization)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                    else:
                        # Fallback to original frame
                        jpeg_bytes = frame_to_jpeg(frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
                else:
                    logger.debug("Frame capture returned None (camera may be busy)")
            except Exception as e:
                logger.error(f"Error in ML feed: {e}", exc_info=True)
            await asyncio.sleep(1/10)  # ~10 fps (ML is slower)
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/api/steering")
async def steering_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time steering direction and angle."""
    await websocket.accept()
    steering_connections.append(websocket)
    
    try:
        while True:
            if not camera_available or detector is None:
                await websocket.send_json({
                    "direction": "straight",
                    "angle": 0,
                    "error": "Camera not available",
                    "method": current_detection_method,
                    "details": {
                        "region_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                        "weighted_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                        "left_score": 0.0,
                        "right_score": 0.0,
                        "total_score": 0.0,
                        "steering_score": 0.0,
                        "total_density": 0.0
                    }
                })
                await asyncio.sleep(1)
                continue
            frame = detector.capture_frame()
            if frame is not None:
                # Use unified lane detector
                steering_data = detector.analyze_lane_position(frame)
                await websocket.send_json(steering_data)
            else:
                # Send default data if frame capture fails
                await websocket.send_json({
                    "direction": "straight",
                    "angle": 0,
                    "method": current_detection_method,
                    "details": {
                        "region_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                        "weighted_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                        "left_score": 0.0,
                        "right_score": 0.0,
                        "total_score": 0.0,
                        "steering_score": 0.0,
                        "total_density": 0.0
                    }
                })
            await asyncio.sleep(0.1)  # 10 updates per second
    except WebSocketDisconnect:
        if websocket in steering_connections:
            steering_connections.remove(websocket)


@app.websocket("/api/logs")
async def logs_websocket(websocket: WebSocket):
    """WebSocket endpoint for log messages."""
    await websocket.accept()
    log_connections.append(websocket)
    
    # Send existing log messages
    for log_entry in log_messages[-100:]:  # Last 100 messages
        try:
            await websocket.send_json(log_entry)
        except:
            break
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in log_connections:
            log_connections.remove(websocket)


@app.get("/api/logs")
async def get_logs():
    """Get recent log messages."""
    return {"logs": log_messages[-100:]}


@app.get("/api/detection_method")
async def get_detection_method():
    """Get current detection method."""
    return {
        "method": current_detection_method,
        "available_methods": ["edge", "fast_scnn", "yolo", "hybrid"]
    }


class DetectionMethodRequest(BaseModel):
    method: str


@app.post("/api/detection_method")
async def set_detection_method(request: DetectionMethodRequest):
    """Set detection method."""
    global detector, current_detection_method
    
    method = request.method
    if method not in ["edge", "fast_scnn", "yolo", "hybrid"]:
        return {"error": f"Invalid method: {method}. Must be one of: edge, fast_scnn, yolo, hybrid"}
    
    try:
        # Cleanup old detector
        if detector is not None:
            detector.cleanup()
        
        # Create new detector with selected method
        from .lane_detector import LaneDetector
        detector = LaneDetector(method=method)
        
        # Reinitialize camera if it was available
        if camera_available:
            detector.initialize_camera(CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX)
        
        current_detection_method = method
        add_log_message("INFO", f"Detection method changed to: {method}")
        
        return {
            "method": current_detection_method,
            "message": f"Detection method set to {method}"
        }
    except Exception as e:
        error_msg = f"Failed to set detection method: {e}"
        add_log_message("ERROR", error_msg)
        logger.error(error_msg)
        return {"error": error_msg}


@app.on_event("startup")
async def startup_event():
    """Initialize camera on startup."""
    global detector, camera_available, current_detection_method
    try:
        detector = LaneDetector(method=current_detection_method)
        detector.initialize_camera(CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX)
        camera_available = True
        add_log_message("INFO", f"Camera initialized successfully with method: {current_detection_method}")
        logger.info(f"Camera initialized successfully with method: {current_detection_method}")
    except Exception as e:
        camera_available = False
        error_msg = f"Failed to initialize camera: {e}"
        add_log_message("ERROR", error_msg)
        logger.error(error_msg)
        logger.warning("Server will continue without camera. Some features will be unavailable.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global detector
    if detector is not None:
        detector.cleanup()
        add_log_message("INFO", "Server shutting down")

