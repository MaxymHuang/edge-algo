"""Shared utilities for steering angle calculation from lane detection."""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

from .config import (
    STEERING_ANGLE_CLASSES,
    MIN_EDGE_DENSITY,
    STEERING_SCORE_THRESHOLD,
    PATH_FOLLOWING_MODE,
    MIN_PATH_DENSITY_THRESHOLD,
    PATH_STEERING_THRESHOLD,
)

logger = logging.getLogger(__name__)


def extract_lane_centerline(lane_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract left and right lane boundaries from segmentation mask.
    
    Args:
        lane_mask: Binary lane segmentation mask (255 for lane, 0 for background)
        
    Returns:
        Tuple of (left_lane_points, right_lane_points) or (None, None) if not found
    """
    if lane_mask is None or lane_mask.size == 0:
        return None, None
    
    h, w = lane_mask.shape
    
    # Focus on lower half of image (where lanes are most visible)
    lower_half = lane_mask[h//2:, :]
    
    # Find lane pixels
    lane_pixels = np.where(lower_half > 128)
    
    if len(lane_pixels[0]) == 0:
        return None, None
    
    # Get x coordinates of lane pixels
    x_coords = lane_pixels[1]
    y_coords = lane_pixels[0] + h//2  # Adjust for lower half offset
    
    # Split into left and right lanes based on x position
    center_x = w // 2
    left_mask = x_coords < center_x
    right_mask = x_coords >= center_x
    
    left_x = x_coords[left_mask]
    left_y = y_coords[left_mask]
    right_x = x_coords[right_mask]
    right_y = y_coords[right_mask]
    
    # Need at least a few points for each lane
    if len(left_x) < 10 or len(right_x) < 10:
        return None, None
    
    # Fit lines to left and right lanes
    try:
        # Use RANSAC or simple line fitting
        left_points = np.column_stack((left_x, left_y))
        right_points = np.column_stack((right_x, right_y))
        
        return left_points, right_points
    except Exception as e:
        logger.warning(f"Failed to extract lane centerline: {e}")
        return None, None


def calculate_steering_from_lanes(
    lane_mask: np.ndarray,
    frame_center_x: Optional[int] = None
) -> Tuple[int, Dict]:
    """
    Calculate steering angle from lane segmentation mask.
    
    Args:
        lane_mask: Binary lane segmentation mask
        frame_center_x: Center x coordinate of frame (default: frame width / 2)
        
    Returns:
        Tuple of (steering_angle, detailed_info_dict)
    """
    if lane_mask is None or lane_mask.size == 0:
        return (0, {
            "error": "No lane mask",
            "steering_score": 0.0,
            "lane_center": None
        })
    
    h, w = lane_mask.shape
    if frame_center_x is None:
        frame_center_x = w // 2
    
    # Extract lane boundaries
    left_points, right_points = extract_lane_centerline(lane_mask)
    
    if left_points is None or right_points is None:
        # Fallback: use density-based approach similar to edge detection
        return calculate_steering_from_mask_density(lane_mask, frame_center_x)
    
    # Calculate lane center at bottom of frame
    # Get average x position of left and right lanes at bottom
    bottom_y = h - 1
    bottom_tolerance = 20  # Look within this many pixels from bottom
    
    left_bottom_x = []
    right_bottom_x = []
    
    for point in left_points:
        if abs(point[1] - bottom_y) < bottom_tolerance:
            left_bottom_x.append(point[0])
    
    for point in right_points:
        if abs(point[1] - bottom_y) < bottom_tolerance:
            right_bottom_x.append(point[0])
    
    if len(left_bottom_x) == 0 or len(right_bottom_x) == 0:
        # Fallback to density method
        return calculate_steering_from_mask_density(lane_mask, frame_center_x)
    
    # Calculate lane center
    left_center_x = np.mean(left_bottom_x)
    right_center_x = np.mean(right_bottom_x)
    lane_center_x = (left_center_x + right_center_x) / 2
    
    # Calculate deviation from frame center
    deviation = lane_center_x - frame_center_x
    max_deviation = w / 2  # Maximum possible deviation
    
    # Normalize to [-1, 1]
    steering_score = deviation / max_deviation
    steering_score = np.clip(steering_score, -1.0, 1.0)
    
    # Map to discrete angle classes
    if abs(steering_score) < STEERING_SCORE_THRESHOLD:
        angle = 0
    else:
        target_angle = steering_score * max(STEERING_ANGLE_CLASSES)
        angle = min(STEERING_ANGLE_CLASSES, key=lambda x: abs(x - target_angle))
    
    return (angle, {
        "steering_score": float(steering_score),
        "lane_center": float(lane_center_x),
        "frame_center": float(frame_center_x),
        "deviation": float(deviation),
        "left_lane_x": float(left_center_x),
        "right_lane_x": float(right_center_x),
        "method": "centerline"
    })


def calculate_steering_from_mask_density(
    lane_mask: np.ndarray,
    frame_center_x: Optional[int] = None
) -> Tuple[int, Dict]:
    """
    Calculate steering angle using density-based approach with path-following logic.
    
    This is a fallback method when centerline extraction fails.
    Uses path-following: steer towards least dense region (path).
    """
    if lane_mask is None or lane_mask.size == 0:
        return (0, {"error": "No lane mask", "steering_score": 0.0})
    
    h, w = lane_mask.shape
    if frame_center_x is None:
        frame_center_x = w // 2
    
    # Divide into five regions (same as edge detection)
    far_left_end = int(w * 0.2)
    left_end = int(w * 0.4)
    center_end = int(w * 0.6)
    right_end = int(w * 0.8)
    
    # Extract regions
    far_left_region = lane_mask[:, :far_left_end]
    left_region = lane_mask[:, far_left_end:left_end]
    center_region = lane_mask[:, left_end:center_end]
    right_region = lane_mask[:, center_end:right_end]
    far_right_region = lane_mask[:, right_end:]
    
    # Calculate lane density for each region
    # Note: For lane masks, higher density = more lane pixels = path
    # But we want to find least dense region for obstacles/edges
    # So we invert: calculate non-lane density (background/obstacles)
    def calc_density(region):
        if region.size == 0:
            return 0.0
        # Calculate non-lane density (obstacles/edges)
        # Lane pixels are > 128, so non-lane pixels are <= 128
        return np.sum(region <= 128) / region.size
    
    far_left_density = calc_density(far_left_region)
    left_density = calc_density(left_region)
    center_density = calc_density(center_region)
    right_density = calc_density(right_region)
    far_right_density = calc_density(far_right_region)
    
    densities = (far_left_density, left_density, center_density, right_density, far_right_density)
    
    # Use path-following mode if enabled
    if PATH_FOLLOWING_MODE:
        # Find the region with minimum density (this is the path)
        min_density = min(densities)
        min_indices = [i for i, d in enumerate(densities) if d == min_density]
        # Prefer center region if tied
        least_dense_idx = 2 if 2 in min_indices else min_indices[0]
        
        # Calculate density difference
        max_density = max(densities)
        density_diff = max_density - min_density
        
        # Check if density difference is significant enough
        if density_diff < PATH_STEERING_THRESHOLD:
            return (0, {
                "steering_score": 0.0,
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
                "method": "path_following"
            })
        
        # Calculate steering direction relative to center
        # Region indices: 0=far_left, 1=left, 2=center, 3=right, 4=far_right
        # Steering score: negative = left, positive = right, 0 = straight
        # Formula: -1.0 + (idx * 0.5) gives: [-1.0, -0.5, 0.0, 0.5, 1.0]
        steering_score = -1.0 + (least_dense_idx * 0.5)
        
        # Scale steering score by density difference
        normalized_diff = min(1.0, density_diff / MIN_PATH_DENSITY_THRESHOLD)
        steering_score = steering_score * normalized_diff
        
        # Map steering score to discrete angle classes
        if abs(steering_score) < STEERING_SCORE_THRESHOLD:
            angle = 0
        else:
            target_angle = steering_score * max(STEERING_ANGLE_CLASSES)
            angle = min(STEERING_ANGLE_CLASSES, key=lambda x: abs(x - target_angle))
        
        return (angle, {
            "steering_score": float(steering_score),
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
            "normalized_diff": float(normalized_diff),
            "method": "path_following"
        })
    
    # Fallback to old obstacle avoidance logic if path-following is disabled
    from .config import REGION_WEIGHTS
    weighted_densities = [
        far_left_density * REGION_WEIGHTS[0],
        left_density * REGION_WEIGHTS[1],
        center_density * REGION_WEIGHTS[2],
        right_density * REGION_WEIGHTS[3],
        far_right_density * REGION_WEIGHTS[4],
    ]
    
    left_score = weighted_densities[0] + weighted_densities[1]
    right_score = weighted_densities[3] + weighted_densities[4]
    
    total_score = left_score + right_score
    total_density = sum(densities) / len(densities)
    
    if total_score == 0 or total_density < MIN_EDGE_DENSITY:
        steering_score = 0.0
        angle = 0
    else:
        steering_score = (left_score - right_score) / total_score
        
        if abs(steering_score) < STEERING_SCORE_THRESHOLD:
            angle = 0
        else:
            target_angle = steering_score * max(STEERING_ANGLE_CLASSES)
            angle = min(STEERING_ANGLE_CLASSES, key=lambda x: abs(x - target_angle))
    
    return (angle, {
        "steering_score": float(steering_score),
        "region_densities": {
            "far_left": float(far_left_density),
            "left": float(left_density),
            "center": float(center_density),
            "right": float(right_density),
            "far_right": float(far_right_density)
        },
        "left_score": float(left_score),
        "right_score": float(right_score),
        "total_density": float(total_density),
        "method": "obstacle_avoidance"
    })
