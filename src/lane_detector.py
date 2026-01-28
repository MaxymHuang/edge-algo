"""Unified lane detection interface supporting edge detection, ML models, and hybrid modes."""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from pathlib import Path

from .edge_detection import EdgeDetector
from .lane_detection_ml import FastSCNNLaneDetector, YOLOLaneDetector
from .steering_utils import calculate_steering_from_lanes
from .config import (
    LANE_DETECTION_METHOD,
    HYBRID_ML_WEIGHT,
    HYBRID_EDGE_WEIGHT,
    ML_MODEL_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class LaneDetector:
    """Unified lane detector supporting multiple detection methods."""
    
    def __init__(self, method: str = None):
        """
        Initialize lane detector.
        
        Args:
            method: Detection method ("edge", "fast_scnn", "yolo", "hybrid")
        """
        self.method = method or LANE_DETECTION_METHOD
        self.edge_detector: Optional[EdgeDetector] = None
        self.fast_scnn: Optional[FastSCNNLaneDetector] = None
        self.yolo: Optional[YOLOLaneDetector] = None
        
        # Initialize based on method
        # Always initialize edge detector for camera access (even if not using edge detection)
        self.edge_detector = EdgeDetector()
        
        if self.method in ["fast_scnn", "hybrid"]:
            try:
                self.fast_scnn = FastSCNNLaneDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize Fast SCNN: {e}")
        
        if self.method in ["yolo", "hybrid"]:
            try:
                self.yolo = YOLOLaneDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize YOLO: {e}")
    
    def initialize_camera(self, resolution: Tuple[int, int] = (640, 480), 
                         framerate: int = 30, camera_index: int = 0):
        """Initialize camera (delegates to edge detector)."""
        if self.edge_detector is not None:
            self.edge_detector.initialize_camera(resolution, framerate, camera_index)
        else:
            logger.error("Edge detector not initialized - this should not happen!")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from camera."""
        if self.edge_detector is not None:
            return self.edge_detector.capture_frame()
        return None
    
    def detect_edges(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect edges using Canny edge detection."""
        if self.edge_detector is not None:
            return self.edge_detector.detect_edges(frame)
        return None
    
    def detect_lanes_ml(self, frame: np.ndarray, model_type: str = "fast_scnn") -> Tuple[Optional[np.ndarray], float, Dict]:
        """
        Detect lanes using ML model.
        
        Args:
            frame: Input frame
            model_type: "fast_scnn" or "yolo"
            
        Returns:
            (segmentation_mask, confidence, metadata)
        """
        if model_type == "fast_scnn" and self.fast_scnn is not None:
            return self.fast_scnn.detect_lanes(frame)
        elif model_type == "yolo" and self.yolo is not None:
            return self.yolo.detect_lanes(frame)
        else:
            return None, 0.0, {"error": f"Model {model_type} not available"}
    
    def analyze_lane_position(self, frame: np.ndarray) -> Dict:
        """
        Analyze lane position and calculate steering angle.
        
        This is the main method that combines all detection methods based on configuration.
        
        Returns:
            Dictionary with direction, angle, and details
        """
        if frame is None:
            return {
                "direction": "straight",
                "angle": 0,
                "details": {"error": "No frame"},
                "method": self.method
            }
        
        if self.method == "edge":
            return self._analyze_edge_only(frame)
        elif self.method == "fast_scnn":
            return self._analyze_fast_scnn_only(frame)
        elif self.method == "yolo":
            return self._analyze_yolo_only(frame)
        elif self.method == "hybrid":
            return self._analyze_hybrid(frame)
        else:
            logger.error(f"Unknown detection method: {self.method}")
            return {
                "direction": "straight",
                "angle": 0,
                "details": {"error": f"Unknown method: {self.method}"},
                "method": self.method
            }
    
    def _analyze_edge_only(self, frame: np.ndarray) -> Dict:
        """Analyze using edge detection only."""
        if self.edge_detector is None:
            return {
                "direction": "straight",
                "angle": 0,
                "details": {"error": "Edge detector not initialized"},
                "method": "edge"
            }
        
        edges = self.edge_detector.detect_edges(frame)
        if edges is None:
            return {
                "direction": "straight",
                "angle": 0,
                "details": {"error": "Edge detection failed"},
                "method": "edge"
            }
        
        result = self.edge_detector.analyze_edge_position(edges)
        result["method"] = "edge"
        return result
    
    def _analyze_fast_scnn_only(self, frame: np.ndarray) -> Dict:
        """Analyze using Fast SCNN only."""
        lane_mask, confidence, metadata = self.detect_lanes_ml(frame, "fast_scnn")
        
        if lane_mask is None or confidence < ML_MODEL_CONFIDENCE_THRESHOLD:
            # Fallback to edge detection if available
            if self.edge_detector is not None:
                logger.debug("Fast SCNN failed, falling back to edge detection")
                return self._analyze_edge_only(frame)
            
            return {
                "direction": "straight",
                "angle": 0,
                "details": {
                    "error": "Fast SCNN detection failed",
                    "confidence": confidence
                },
                "method": "fast_scnn"
            }
        
        # Calculate steering from lane mask
        angle, angle_details = calculate_steering_from_lanes(lane_mask)
        
        # Determine direction
        if angle == 0:
            direction = "straight"
        elif angle < 0:
            direction = "left"
        else:
            direction = "right"
        
        return {
            "direction": direction,
            "angle": angle,
            "details": {
                **angle_details,
                "ml_confidence": confidence,
                **metadata
            },
            "method": "fast_scnn"
        }
    
    def _analyze_yolo_only(self, frame: np.ndarray) -> Dict:
        """Analyze using YOLO only."""
        lane_mask, confidence, metadata = self.detect_lanes_ml(frame, "yolo")
        
        if lane_mask is None or confidence < ML_MODEL_CONFIDENCE_THRESHOLD:
            # Fallback to edge detection if available
            if self.edge_detector is not None:
                logger.debug("YOLO failed, falling back to edge detection")
                return self._analyze_edge_only(frame)
            
            return {
                "direction": "straight",
                "angle": 0,
                "details": {
                    "error": "YOLO detection failed",
                    "confidence": confidence
                },
                "method": "yolo"
            }
        
        # Calculate steering from lane mask
        angle, angle_details = calculate_steering_from_lanes(lane_mask)
        
        # Determine direction
        if angle == 0:
            direction = "straight"
        elif angle < 0:
            direction = "left"
        else:
            direction = "right"
        
        return {
            "direction": direction,
            "angle": angle,
            "details": {
                **angle_details,
                "ml_confidence": confidence,
                **metadata
            },
            "method": "yolo"
        }
    
    def _analyze_hybrid(self, frame: np.ndarray) -> Dict:
        """Analyze using hybrid approach (ML + edge detection)."""
        ml_angle = 0
        ml_confidence = 0.0
        ml_details = {}
        ml_method = None
        
        # Try ML detection first
        if self.fast_scnn is not None:
            lane_mask, confidence, metadata = self.fast_scnn.detect_lanes(frame)
            if lane_mask is not None and confidence >= ML_MODEL_CONFIDENCE_THRESHOLD:
                ml_angle, ml_details = calculate_steering_from_lanes(lane_mask)
                ml_confidence = confidence
                ml_method = "fast_scnn"
                ml_details.update(metadata)
        elif self.yolo is not None:
            lane_mask, confidence, metadata = self.yolo.detect_lanes(frame)
            if lane_mask is not None and confidence >= ML_MODEL_CONFIDENCE_THRESHOLD:
                ml_angle, ml_details = calculate_steering_from_lanes(lane_mask)
                ml_confidence = confidence
                ml_method = "yolo"
                ml_details.update(metadata)
        
        # Get edge detection result
        edge_angle = 0
        edge_details = {}
        if self.edge_detector is not None:
            edges = self.edge_detector.detect_edges(frame)
            if edges is not None:
                edge_result = self.edge_detector.analyze_edge_position(edges)
                edge_angle = edge_result.get("angle", 0)
                edge_details = edge_result.get("details", {})
        
        # Combine results
        if ml_confidence >= ML_MODEL_CONFIDENCE_THRESHOLD:
            # Weighted combination
            combined_angle = int(
                ml_angle * HYBRID_ML_WEIGHT + edge_angle * HYBRID_EDGE_WEIGHT
            )
            # Map to nearest angle class
            from .config import STEERING_ANGLE_CLASSES
            combined_angle = min(STEERING_ANGLE_CLASSES, 
                               key=lambda x: abs(x - combined_angle))
        else:
            # Low ML confidence, use edge detection
            combined_angle = edge_angle
        
        # Determine direction
        if combined_angle == 0:
            direction = "straight"
        elif combined_angle < 0:
            direction = "left"
        else:
            direction = "right"
        
        return {
            "direction": direction,
            "angle": combined_angle,
            "details": {
                "ml_angle": ml_angle,
                "ml_confidence": ml_confidence,
                "ml_method": ml_method,
                "edge_angle": edge_angle,
                "combined_angle": combined_angle,
                "ml_weight": HYBRID_ML_WEIGHT,
                "edge_weight": HYBRID_EDGE_WEIGHT,
                **ml_details,
                "edge_details": edge_details
            },
            "method": "hybrid"
        }
    
    def get_edge_visualization(self, frame: np.ndarray, edges: np.ndarray = None, 
                              edges_only: bool = False) -> Optional[np.ndarray]:
        """Get edge detection visualization."""
        if self.edge_detector is not None:
            if edges is None:
                edges = self.edge_detector.detect_edges(frame)
            if edges is not None:
                return self.edge_detector.get_edge_visualization(frame, edges, edges_only)
        return None
    
    def get_ml_visualization(self, frame: np.ndarray, lane_mask: np.ndarray = None,
                             model_type: str = "fast_scnn") -> Optional[np.ndarray]:
        """
        Get ML segmentation visualization overlay.
        
        Args:
            frame: Original frame
            lane_mask: Lane segmentation mask (if None, will detect)
            model_type: "fast_scnn" or "yolo"
            
        Returns:
            Visualization with lane mask overlaid
        """
        if lane_mask is None:
            lane_mask, _, _ = self.detect_lanes_ml(frame, model_type)
        
        if lane_mask is None:
            return frame.copy()
        
        # Create overlay
        result = frame.copy()
        
        # Create colored mask (green for lanes)
        mask_colored = np.zeros_like(result)
        mask_colored[lane_mask > 128] = [0, 255, 0]  # Green
        
        # Blend with original
        overlay = cv2.addWeighted(result, 0.7, mask_colored, 0.3, 0)
        
        return overlay
    
    def cleanup(self):
        """Clean up all resources."""
        if self.edge_detector is not None:
            self.edge_detector.cleanup()
        
        if self.fast_scnn is not None:
            self.fast_scnn.unload_model()
        
        if self.yolo is not None:
            self.yolo.unload_model()
    
    @staticmethod
    def create(method: str = None) -> 'LaneDetector':
        """
        Factory method to create lane detector.
        
        Args:
            method: Detection method ("edge", "fast_scnn", "yolo", "hybrid")
            
        Returns:
            LaneDetector instance
        """
        return LaneDetector(method=method)
