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
    PATH_FOLLOWING_MODE,
    MIN_PATH_DENSITY_THRESHOLD,
    PATH_STEERING_THRESHOLD,
    SPARSE_EDGE_MODE,
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
        """Apply enhanced Canny edge detection with Sobel fallback for sparse scenarios."""
        if frame is None:
            return None
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # Apply contrast enhancement for sparse edge scenarios
        if SPARSE_EDGE_MODE:
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to improve edge detection
        # Calculate median of the image for adaptive thresholds
        median = np.median(blurred)
        # More aggressive thresholds for sparse scenarios
        if SPARSE_EDGE_MODE:
            lower = int(max(0, 0.5 * median))  # More aggressive lower threshold
            upper = int(min(255, 1.5 * median))
        else:
            lower = int(max(0, 0.7 * median))
            upper = int(min(255, 1.3 * median))
        
        # Use adaptive thresholds or fallback to config values
        low_thresh = max(CANNY_LOW_THRESHOLD, lower)
        high_thresh = min(CANNY_HIGH_THRESHOLD, upper)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Fallback to Sobel operator if too few edges detected (sparse scenario)
        if SPARSE_EDGE_MODE:
            edge_count = np.sum(edges > 0)
            total_pixels = edges.size
            edge_density = edge_count / total_pixels if total_pixels > 0 else 0.0
            
            # If edge density is very low, try Sobel as fallback
            if edge_density < 0.001:  # Less than 0.1% edges detected
                # Apply Sobel operator
                sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
                
                # Threshold Sobel result
                _, sobel_edges = cv2.threshold(sobel_magnitude, 30, 255, cv2.THRESH_BINARY)
                
                # Combine Canny and Sobel results
                edges = cv2.bitwise_or(edges, sobel_edges)
        
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
    
    def _find_least_dense_region(self, densities: Tuple[float, float, float, float, float]) -> int:
        """
        Find the region with minimum edge density (the path).
        
        Args:
            densities: Tuple of (far_left, left, center, right, far_right) densities
            
        Returns:
            Index of region with minimum density (0=far_left, 1=left, 2=center, 3=right, 4=far_right)
        """
        min_density = min(densities)
        # Find all regions with minimum density (handle ties)
        min_indices = [i for i, d in enumerate(densities) if d == min_density]
        # If multiple regions have same minimum, prefer center region
        if 2 in min_indices:
            return 2
        # Otherwise return the first minimum index
        return min_indices[0]
    
    def _calculate_path_steering(
        self, 
        least_dense_idx: int, 
        densities: Tuple[float, float, float, float, float]
    ) -> Tuple[int, Dict[str, any]]:
        """
        Calculate steering angle to move towards the least dense region (path).
        
        Args:
            least_dense_idx: Index of region with minimum density (0-4)
            densities: Tuple of region densities
            
        Returns:
            Tuple of (steering_angle, detailed_info_dict)
        """
        far_left_density, left_density, center_density, right_density, far_right_density = densities
        
        # Camera center is at region index 2 (center region: 40-60% of width)
        CENTER_REGION_IDX = 2
        
        # Calculate density difference to determine steering magnitude
        min_density = densities[least_dense_idx]
        max_density = max(densities)
        density_diff = max_density - min_density
        
        # Check if density difference is significant enough
        if density_diff < PATH_STEERING_THRESHOLD:
            # Not enough difference, steer straight
            return (0, {
                "region_densities": {
                    "far_left": float(far_left_density),
                    "left": float(left_density),
                    "center": float(center_density),
                    "right": float(right_density),
                    "far_right": float(far_right_density)
                },
                "least_dense_region": least_dense_idx,
                "min_density": float(min_density),
                "max_density": float(max_density),
                "density_diff": float(density_diff),
                "steering_score": 0.0,
                "method": "path_following"
            })
        
        # Calculate steering direction relative to center
        # Region indices: 0=far_left, 1=left, 2=center, 3=right, 4=far_right
        # Steering score: negative = left, positive = right, 0 = straight
        # Formula: -1.0 + (idx * 0.5) gives: [-1.0, -0.5, 0.0, 0.5, 1.0]
        steering_score = -1.0 + (least_dense_idx * 0.5)
        
        # Scale steering score by density difference (stronger difference = stronger steering)
        # Normalize density_diff to [0, 1] range for scaling
        normalized_diff = min(1.0, density_diff / MIN_PATH_DENSITY_THRESHOLD)
        steering_score = steering_score * normalized_diff
        
        # Map steering score to discrete angle classes
        if abs(steering_score) < STEERING_SCORE_THRESHOLD:
            angle = 0
        else:
            target_angle = steering_score * max(STEERING_ANGLE_CLASSES)
            angle = min(STEERING_ANGLE_CLASSES, key=lambda x: abs(x - target_angle))
        
        return (angle, {
            "region_densities": {
                "far_left": float(far_left_density),
                "left": float(left_density),
                "center": float(center_density),
                "right": float(right_density),
                "far_right": float(far_right_density)
            },
            "least_dense_region": least_dense_idx,
            "min_density": float(min_density),
            "max_density": float(max_density),
            "density_diff": float(density_diff),
            "steering_score": float(steering_score),
            "normalized_diff": float(normalized_diff),
            "method": "path_following"
        })
    
    def calculate_steering_angle(self, edges: np.ndarray) -> Tuple[int, Dict[str, any]]:
        """
        Calculate steering angle based on path-following: steer towards least dense region.
        
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
                "least_dense_region": 2,
                "min_density": 0.0,
                "max_density": 0.0,
                "density_diff": 0.0,
                "steering_score": 0.0,
                "method": "path_following"
            })
        
        # Calculate densities for each region
        densities = self.calculate_region_densities(edges)
        far_left_density, left_density, center_density, right_density, far_right_density = densities
        
        # Use path-following mode if enabled
        if PATH_FOLLOWING_MODE:
            # Find the region with minimum density (this is the path)
            least_dense_idx = self._find_least_dense_region(densities)
            
            # Calculate steering to move towards the least dense region
            return self._calculate_path_steering(least_dense_idx, densities)
        
        # Fallback to old obstacle avoidance logic if path-following is disabled
        # (for backward compatibility)
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
        left_score = weighted_densities[0] + weighted_densities[1]
        right_score = weighted_densities[3] + weighted_densities[4]
        
        total_score = left_score + right_score
        
        if total_score == 0:
            steering_score = 0.0
            angle = 0
        else:
            steering_score = (left_score - right_score) / total_score
            
            if total_density < MIN_EDGE_DENSITY:
                angle = 0
            elif abs(steering_score) < STEERING_SCORE_THRESHOLD:
                angle = 0
            else:
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
            "total_density": float(total_density),
            "method": "obstacle_avoidance"
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

