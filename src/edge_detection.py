"""Core edge detection and steering analysis module."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import sys
import os

# Add system dist-packages to path for libcamera access
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError as e:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None
    import_error = e

from .config import (
    CANNY_LOW_THRESHOLD,
    CANNY_HIGH_THRESHOLD,
    EDGE_DENSITY_THRESHOLD,
    MIN_EDGE_DENSITY,
    STEERING_ANGLE_CLASSES,
    REGION_WEIGHTS,
    ANGLE_DENSITY_THRESHOLD,
    STEERING_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class EdgeDetector:
    """Edge detection and steering analysis class."""
    
    def __init__(self):
        self.camera: Optional[Picamera2] = None
        self.is_initialized = False
    
    def list_cameras(self):
        """List available cameras."""
        if not PICAMERA2_AVAILABLE:
            return []
        
        try:
            import libcamera
            camera_manager = libcamera.CameraManager.singleton()
            cameras = []
            for i, camera in enumerate(camera_manager.cameras):
                cameras.append({
                    "index": i,
                    "id": camera.id,
                    "model": camera.properties.get("Model", "Unknown")
                })
            return cameras
        except Exception as e:
            logger.warning(f"Failed to list cameras: {e}")
            return []
    
    def initialize_camera(self, resolution: Tuple[int, int] = (640, 480), framerate: int = 30, camera_index: int = 0):
        """Initialize the Raspberry Pi Camera Module.
        
        Args:
            resolution: Camera resolution (width, height)
            framerate: Camera framerate
            camera_index: Camera index (0 for Cam0, 1 for Cam1, etc.)
        """
        if not PICAMERA2_AVAILABLE:
            error_msg = (
                f"picamera2 is not available. This usually means libcamera is not installed.\n"
                f"On Raspberry Pi OS, install it with: sudo apt update && sudo apt install -y python3-libcamera\n"
                f"Original error: {import_error}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # List available cameras for debugging
        available_cameras = self.list_cameras()
        if available_cameras:
            cam_list = ", ".join([f"Cam{cam['index']}: {cam['id']}" for cam in available_cameras])
            logger.info(f"Available cameras: {cam_list}")
        else:
            logger.warning("No cameras detected or unable to list cameras")
        
        try:
            # Initialize camera with specified index
            self.camera = Picamera2(camera_num=camera_index)
            config = self.camera.create_preview_configuration(
                main={"size": resolution, "format": "RGB888"},
                controls={"FrameRate": framerate}
            )
            self.camera.configure(config)
            self.camera.start()
            self.is_initialized = True
            
            # Get camera info
            camera_info = self.camera.camera_properties
            camera_model = camera_info.get("Model", "Unknown")
            logger.info(f"Camera initialized: Cam{camera_index} ({camera_model}) with resolution {resolution} at {framerate} fps")
        except Exception as e:
            error_msg = f"Failed to initialize camera Cam{camera_index}: {e}"
            if available_cameras:
                cam_list = ", ".join([f"Cam{cam['index']}" for cam in available_cameras])
                error_msg += f"\nAvailable cameras: {cam_list}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera."""
        if not self.is_initialized or self.camera is None:
            logger.error("Camera not initialized")
            return None
        
        try:
            frame = self.camera.capture_array()
            return frame
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection to the frame."""
        if frame is None:
            return None
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to improve edge detection
        # Calculate median of the image for adaptive thresholds
        median = np.median(blurred)
        lower = int(max(0, 0.7 * median))
        upper = int(min(255, 1.3 * median))
        
        # Use adaptive thresholds or fallback to config values
        low_thresh = max(CANNY_LOW_THRESHOLD, lower)
        high_thresh = min(CANNY_HIGH_THRESHOLD, upper)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        return edges
    
    def calculate_region_densities(self, edges: np.ndarray) -> Tuple[float, float, float, float, float]:
        """
        Calculate edge density for five horizontal regions of the frame.
        
        Regions:
        - Far-left: 0-20% of frame width
        - Left: 20-40% of frame width
        - Center: 40-60% of frame width
        - Right: 60-80% of frame width
        - Far-right: 80-100% of frame width
        
        Returns:
            Tuple of (far_left_density, left_density, center_density, right_density, far_right_density)
        """
        if edges is None:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        
        height, width = edges.shape
        
        # Calculate region boundaries
        far_left_end = int(width * 0.2)
        left_end = int(width * 0.4)
        center_end = int(width * 0.6)
        right_end = int(width * 0.8)
        
        # Extract regions
        far_left_region = edges[:, :far_left_end]
        left_region = edges[:, far_left_end:left_end]
        center_region = edges[:, left_end:center_end]
        right_region = edges[:, center_end:right_end]
        far_right_region = edges[:, right_end:]
        
        # Calculate edge density for each region (proportion of edge pixels)
        far_left_density = np.sum(far_left_region > 0) / far_left_region.size if far_left_region.size > 0 else 0.0
        left_density = np.sum(left_region > 0) / left_region.size if left_region.size > 0 else 0.0
        center_density = np.sum(center_region > 0) / center_region.size if center_region.size > 0 else 0.0
        right_density = np.sum(right_region > 0) / right_region.size if right_region.size > 0 else 0.0
        far_right_density = np.sum(far_right_region > 0) / far_right_region.size if far_right_region.size > 0 else 0.0
        
        return (far_left_density, left_density, center_density, right_density, far_right_density)
    
    def calculate_steering_angle(self, edges: np.ndarray) -> Tuple[int, Dict[str, any]]:
        """
        Calculate steering angle based on weighted edge density distribution.
        
        Args:
            edges: Edge detection result
            
        Returns:
            Tuple of (steering_angle, detailed_info_dict)
            - steering_angle: Steering angle in degrees (from STEERING_ANGLE_CLASSES)
            - detailed_info_dict: Dictionary with calculation details
        """
        if edges is None:
            return (0, {
                "region_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                "weighted_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                "left_score": 0.0,
                "right_score": 0.0,
                "total_score": 0.0,
                "steering_score": 0.0,
                "total_density": 0.0
            })
        
        # Calculate densities for each region
        densities = self.calculate_region_densities(edges)
        far_left_density, left_density, center_density, right_density, far_right_density = densities
        
        # Check if we have enough edges to make a decision
        total_density = sum(densities) / len(densities)
        
        # Apply weights to densities
        weighted_densities = [
            far_left_density * REGION_WEIGHTS[0],
            left_density * REGION_WEIGHTS[1],
            center_density * REGION_WEIGHTS[2],
            right_density * REGION_WEIGHTS[3],
            far_right_density * REGION_WEIGHTS[4],
        ]
        
        # Calculate weighted left and right scores
        # Left side: far-left and left regions
        left_score = weighted_densities[0] + weighted_densities[1]
        # Right side: right and far-right regions
        right_score = weighted_densities[3] + weighted_densities[4]
        
        # Calculate steering score (normalized difference)
        total_score = left_score + right_score
        
        # Normalized score: -1 (hard left) to +1 (hard right)
        if total_score == 0:
            steering_score = 0.0
            angle = 0
        else:
            steering_score = (right_score - left_score) / total_score
            
            # Check if we have enough edges to make a decision
            if total_density < MIN_EDGE_DENSITY:
                angle = 0  # Straight
            # Check if difference is significant enough
            elif abs(steering_score) < STEERING_SCORE_THRESHOLD:
                angle = 0  # Straight
            else:
                # Map steering score to discrete angle classes
                # Find the closest angle class
                angle = min(STEERING_ANGLE_CLASSES, key=lambda x: abs(x - (steering_score * max(STEERING_ANGLE_CLASSES))))
        
        detailed_info = {
            "region_densities": {
                "far_left": float(far_left_density),
                "left": float(left_density),
                "center": float(center_density),
                "right": float(right_density),
                "far_right": float(far_right_density)
            },
            "weighted_densities": {
                "far_left": float(weighted_densities[0]),
                "left": float(weighted_densities[1]),
                "center": float(weighted_densities[2]),
                "right": float(weighted_densities[3]),
                "far_right": float(weighted_densities[4])
            },
            "left_score": float(left_score),
            "right_score": float(right_score),
            "total_score": float(total_score),
            "steering_score": float(steering_score),
            "total_density": float(total_density)
        }
        
        return (angle, detailed_info)
    
    def analyze_edge_position(self, edges: np.ndarray) -> Dict[str, any]:
        """
        Analyze edge positions to determine steering direction and angle.
        
        Returns:
            Dictionary with:
            - "direction": "left", "right", or "straight"
            - "angle": steering angle in degrees
            - "details": detailed calculation results including region densities, scores, etc.
        """
        if edges is None:
            return {
                "direction": "straight",
                "angle": 0,
                "details": {
                    "region_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                    "weighted_densities": {"far_left": 0.0, "left": 0.0, "center": 0.0, "right": 0.0, "far_right": 0.0},
                    "left_score": 0.0,
                    "right_score": 0.0,
                    "total_score": 0.0,
                    "steering_score": 0.0,
                    "total_density": 0.0
                }
            }
        
        # Calculate steering angle with detailed information
        angle, detailed_info = self.calculate_steering_angle(edges)
        
        # Determine direction string for backward compatibility
        if angle == 0:
            direction = "straight"
        elif angle < 0:
            direction = "left"
        else:
            direction = "right"
        
        return {
            "direction": direction,
            "angle": angle,
            "details": detailed_info
        }
    
    def get_edge_visualization(self, frame: np.ndarray, edges: np.ndarray, edges_only: bool = False) -> np.ndarray:
        """Create visualization combining original frame with edge detection overlay.
        
        Args:
            frame: Original camera frame
            edges: Edge detection result
            edges_only: If True, show only edges on black background. If False, overlay edges on original frame.
        """
        if frame is None or edges is None:
            return None
        
        if edges_only:
            # Show edges-only view on black background
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            # Make edges bright cyan/yellow for better visibility
            edges_colored[edges > 0] = [0, 255, 255]  # Cyan color
            # Dilate slightly to make thin edges more visible
            kernel = np.ones((2, 2), np.uint8)
            edges_dilated = cv2.dilate(edges_colored, kernel, iterations=1)
            return edges_dilated
        else:
            # Overlay edges on original frame
            result = frame.copy()
            
            # Dilate edges slightly to make them more visible
            kernel = np.ones((2, 2), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Draw edges in bright cyan for better visibility
            result[edges_dilated > 0] = [0, 255, 255]  # Cyan color
            
            return result
    
    def cleanup(self):
        """Clean up camera resources."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
                self.is_initialized = False
                logger.info("Camera cleaned up")
            except Exception as e:
                logger.error(f"Error during camera cleanup: {e}")

