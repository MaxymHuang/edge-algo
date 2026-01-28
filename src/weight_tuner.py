"""Bayesian optimization for steering weight tuning."""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import logging
import optuna
from collections import deque

from .config import STEERING_ANGLE_CLASSES, MIN_EDGE_DENSITY, STEERING_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class SteeringSample:
    """A single sample of edge densities and human steering input."""
    densities: Tuple[float, float, float, float, float]  # far_left, left, center, right, far_right
    human_angle: int  # Human-provided steering angle


class SteeringDataCollector:
    """Collects and buffers steering training data."""
    
    def __init__(self, buffer_size: int = 500):
        """
        Initialize the data collector.
        
        Args:
            buffer_size: Maximum number of samples to keep in buffer
        """
        self.buffer: deque[SteeringSample] = deque(maxlen=buffer_size)
        self.current_human_angle: int = 0
    
    def set_human_angle(self, angle: int):
        """Set the current human steering angle."""
        if angle in STEERING_ANGLE_CLASSES:
            self.current_human_angle = angle
        else:
            # Find closest valid angle
            self.current_human_angle = min(STEERING_ANGLE_CLASSES, key=lambda x: abs(x - angle))
    
    def add_sample(self, densities: Tuple[float, float, float, float, float]):
        """
        Add a new sample with the current human angle.
        
        Args:
            densities: Tuple of (far_left, left, center, right, far_right) densities
        """
        sample = SteeringSample(densities=densities, human_angle=self.current_human_angle)
        self.buffer.append(sample)
    
    def get_samples(self) -> List[SteeringSample]:
        """Get all samples in the buffer."""
        return list(self.buffer)
    
    def get_sample_count(self) -> int:
        """Get the number of samples in the buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all samples from the buffer."""
        self.buffer.clear()


def calculate_steering_with_weights(
    densities: Tuple[float, float, float, float, float],
    weights: List[float]
) -> int:
    """
    Calculate steering angle using given weights.
    
    Args:
        densities: Tuple of (far_left, left, center, right, far_right) densities
        weights: List of 5 weights for each region
        
    Returns:
        Predicted steering angle
    """
    far_left, left, center, right, far_right = densities
    
    # Apply weights
    weighted = [
        far_left * weights[0],
        left * weights[1],
        center * weights[2],
        right * weights[3],
        far_right * weights[4],
    ]
    
    # Calculate scores
    left_score = weighted[0] + weighted[1]
    right_score = weighted[3] + weighted[4]
    total_score = left_score + right_score
    
    # Calculate average density
    total_density = sum(densities) / len(densities)
    
    if total_score == 0 or total_density < MIN_EDGE_DENSITY:
        return 0
    
    # High left density → steer right (positive) to center
    # High right density → steer left (negative) to center
    steering_score = (left_score - right_score) / total_score
    
    if abs(steering_score) < STEERING_SCORE_THRESHOLD:
        return 0
    
    # Map to discrete angle
    angle = min(STEERING_ANGLE_CLASSES, key=lambda x: abs(x - (steering_score * max(STEERING_ANGLE_CLASSES))))
    return angle


class WeightOptimizer:
    """Bayesian optimizer for steering weights using Optuna."""
    
    def __init__(
        self,
        data_collector: SteeringDataCollector,
        n_trials: int = 50,
        outer_weight_range: Tuple[float, float] = (1.0, 2.0),
        inner_weight_range: Tuple[float, float] = (0.8, 1.5),
        center_weight_range: Tuple[float, float] = (0.5, 1.2),
    ):
        """
        Initialize the weight optimizer.
        
        Args:
            data_collector: Data collector with training samples
            n_trials: Number of optimization trials
            outer_weight_range: Search range for outer region weights (far-left, far-right)
            inner_weight_range: Search range for inner region weights (left, right)
            center_weight_range: Search range for center weight
        """
        self.data_collector = data_collector
        self.n_trials = n_trials
        self.outer_weight_range = outer_weight_range
        self.inner_weight_range = inner_weight_range
        self.center_weight_range = center_weight_range
        
        self.study: Optional[optuna.Study] = None
        self.best_weights: Optional[List[float]] = None
        self.best_mae: Optional[float] = None
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Mean absolute error between predicted and human angles
        """
        # Sample weights with symmetry constraint
        outer_weight = trial.suggest_float("outer_weight", *self.outer_weight_range)
        inner_weight = trial.suggest_float("inner_weight", *self.inner_weight_range)
        center_weight = trial.suggest_float("center_weight", *self.center_weight_range)
        
        # Symmetric weights: [outer, inner, center, inner, outer]
        weights = [outer_weight, inner_weight, center_weight, inner_weight, outer_weight]
        
        samples = self.data_collector.get_samples()
        if not samples:
            return float('inf')
        
        # Calculate MAE
        errors = []
        for sample in samples:
            predicted = calculate_steering_with_weights(sample.densities, weights)
            error = abs(predicted - sample.human_angle)
            errors.append(error)
        
        mae = np.mean(errors)
        return mae
    
    def optimize(self, verbose: bool = True) -> Tuple[List[float], float]:
        """
        Run the optimization.
        
        Args:
            verbose: Whether to show optimization progress
            
        Returns:
            Tuple of (best_weights, best_mae)
        """
        sample_count = self.data_collector.get_sample_count()
        if sample_count < 10:
            logger.warning(f"Only {sample_count} samples available. Need at least 10 for reliable optimization.")
            if sample_count == 0:
                return [1.5, 1.2, 0.8, 1.2, 1.5], float('inf')
        
        # Create study
        optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)
        
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        
        self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=verbose)
        
        # Extract best weights
        best_trial = self.study.best_trial
        outer = best_trial.params["outer_weight"]
        inner = best_trial.params["inner_weight"]
        center = best_trial.params["center_weight"]
        
        self.best_weights = [outer, inner, center, inner, outer]
        self.best_mae = best_trial.value
        
        logger.info(f"Optimization complete. Best MAE: {self.best_mae:.2f}°")
        logger.info(f"Best weights: outer={outer:.3f}, inner={inner:.3f}, center={center:.3f}")
        
        return self.best_weights, self.best_mae
    
    def get_current_best(self) -> Optional[Dict]:
        """Get current best result if optimization has been run."""
        if self.best_weights is None:
            return None
        
        return {
            "weights": self.best_weights,
            "mae": self.best_mae,
            "outer": self.best_weights[0],
            "inner": self.best_weights[1],
            "center": self.best_weights[2],
        }


def save_weights_to_config(weights: List[float], config_path: str = None) -> bool:
    """
    Save optimized weights back to config.py.
    
    Args:
        weights: List of 5 weights [outer, inner, center, inner, outer]
        config_path: Path to config.py (defaults to src/config.py)
        
    Returns:
        True if successful, False otherwise
    """
    import os
    
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Find and replace REGION_WEIGHTS line
        import re
        pattern = r'REGION_WEIGHTS\s*=\s*\[[\d.,\s]+\]'
        weights_str = f"REGION_WEIGHTS = [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}, {weights[3]:.3f}, {weights[4]:.3f}]"
        
        new_content, count = re.subn(pattern, weights_str, content)
        
        if count == 0:
            logger.error("Could not find REGION_WEIGHTS in config.py")
            return False
        
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Saved weights to {config_path}: {weights}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save weights to config: {e}")
        return False

