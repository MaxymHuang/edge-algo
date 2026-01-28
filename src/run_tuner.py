#!/usr/bin/env python3
"""
Steering Weight Tuner - Bayesian optimization for REGION_WEIGHTS.

Usage:
    uv run python -m src.run_tuner

Controls:
    Left Arrow  : Steer left (-45° to -15°)
    Right Arrow : Steer right (+15° to +45°)
    Up Arrow    : Straight (0°)
    1-7         : Set specific angle class
    O           : Run optimization with current samples
    S           : Save best weights to config
    R           : Reset/clear samples
    Q           : Quit
"""

import sys
import os
import logging
import time
import select
import termios
import tty
from typing import Optional

# Add system dist-packages for libcamera
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

from .edge_detection import EdgeDetector
from .config import CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX, STEERING_ANGLE_CLASSES, REGION_WEIGHTS
from .weight_tuner import SteeringDataCollector, WeightOptimizer, save_weights_to_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeyboardInput:
    """Non-blocking keyboard input handler for terminal."""
    
    def __init__(self):
        self.old_settings = None
        self.fd = sys.stdin.fileno()
    
    def __enter__(self):
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self
    
    def __exit__(self, *args):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self) -> Optional[str]:
        """Get a key press if available, non-blocking."""
        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            # Handle arrow keys (escape sequences)
            if char == '\x1b':
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char += sys.stdin.read(2)
            return char
        return None


def print_status(
    collector: SteeringDataCollector,
    current_angle: int,
    predicted_angle: int,
    densities: tuple,
    best_result: Optional[dict] = None
):
    """Print current status to terminal."""
    # Clear line and print status
    status_parts = [
        f"Human: {current_angle:+3d}°",
        f"Pred: {predicted_angle:+3d}°",
        f"Samples: {collector.get_sample_count():4d}",
    ]
    
    if best_result:
        status_parts.append(f"Best MAE: {best_result['mae']:.1f}°")
    
    # Density bar visualization
    density_bar = ""
    chars = "▁▂▃▄▅▆▇█"
    for d in densities:
        idx = min(int(d * len(chars) * 10), len(chars) - 1)
        density_bar += chars[idx]
    status_parts.append(f"[{density_bar}]")
    
    print(f"\r{' | '.join(status_parts):<80}", end="", flush=True)


def print_help():
    """Print control help."""
    print("\n" + "=" * 60)
    print("Steering Weight Tuner - Bayesian Optimization")
    print("=" * 60)
    print("Controls:")
    print("  ← / →     : Steer left/right (increments of 15°)")
    print("  ↑         : Straight (0°)")
    print("  1-7       : Direct angle selection")
    for i, angle in enumerate(STEERING_ANGLE_CLASSES):
        print(f"              {i+1} = {angle:+d}°")
    print("  O         : Run optimization")
    print("  S         : Save best weights to config")
    print("  R         : Reset samples")
    print("  H         : Show this help")
    print("  Q         : Quit")
    print("=" * 60)
    print(f"Current weights: {REGION_WEIGHTS}")
    print("=" * 60 + "\n")


def main():
    """Main tuning loop."""
    print_help()
    
    # Initialize components
    detector = EdgeDetector()
    collector = SteeringDataCollector(buffer_size=1000)
    optimizer = WeightOptimizer(collector, n_trials=30)
    
    best_result = None
    current_human_angle = 0
    collector.set_human_angle(current_human_angle)
    
    # Initialize camera
    try:
        detector.initialize_camera(CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX)
        logger.info("Camera initialized. Start driving and use arrow keys to provide steering input.")
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        logger.info("Running in demo mode without camera...")
        return
    
    try:
        with KeyboardInput() as keyboard:
            running = True
            last_sample_time = 0
            sample_interval = 0.1  # 10 samples per second
            
            print("\nReady! Use arrow keys to steer. Press 'H' for help.\n")
            
            while running:
                # Check for key press
                key = keyboard.get_key()
                
                if key:
                    if key == 'q' or key == 'Q':
                        print("\n\nQuitting...")
                        running = False
                        
                    elif key == '\x1b[D':  # Left arrow
                        # Decrease angle (more left)
                        idx = STEERING_ANGLE_CLASSES.index(current_human_angle) if current_human_angle in STEERING_ANGLE_CLASSES else 3
                        idx = max(0, idx - 1)
                        current_human_angle = STEERING_ANGLE_CLASSES[idx]
                        collector.set_human_angle(current_human_angle)
                        
                    elif key == '\x1b[C':  # Right arrow
                        # Increase angle (more right)
                        idx = STEERING_ANGLE_CLASSES.index(current_human_angle) if current_human_angle in STEERING_ANGLE_CLASSES else 3
                        idx = min(len(STEERING_ANGLE_CLASSES) - 1, idx + 1)
                        current_human_angle = STEERING_ANGLE_CLASSES[idx]
                        collector.set_human_angle(current_human_angle)
                        
                    elif key == '\x1b[A':  # Up arrow
                        current_human_angle = 0
                        collector.set_human_angle(current_human_angle)
                        
                    elif key in '1234567':
                        idx = int(key) - 1
                        if idx < len(STEERING_ANGLE_CLASSES):
                            current_human_angle = STEERING_ANGLE_CLASSES[idx]
                            collector.set_human_angle(current_human_angle)
                    
                    elif key == 'o' or key == 'O':
                        print("\n\nRunning optimization...")
                        weights, mae = optimizer.optimize(verbose=True)
                        best_result = optimizer.get_current_best()
                        print(f"\nOptimization complete!")
                        print(f"Best weights: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}, {weights[3]:.3f}, {weights[4]:.3f}]")
                        print(f"Best MAE: {mae:.2f}°\n")
                        
                    elif key == 's' or key == 'S':
                        if best_result:
                            print("\n\nSaving weights to config...")
                            if save_weights_to_config(best_result['weights']):
                                print("Weights saved successfully!")
                            else:
                                print("Failed to save weights.")
                            print()
                        else:
                            print("\n\nNo optimization result to save. Run optimization first (press 'O').\n")
                    
                    elif key == 'r' or key == 'R':
                        collector.clear()
                        best_result = None
                        print("\n\nSamples cleared.\n")
                    
                    elif key == 'h' or key == 'H':
                        print_help()
                
                # Capture and process frame
                current_time = time.time()
                if current_time - last_sample_time >= sample_interval:
                    frame = detector.capture_frame()
                    if frame is not None:
                        edges = detector.detect_edges(frame)
                        if edges is not None:
                            # Get densities
                            densities = detector.calculate_region_densities(edges)
                            
                            # Add sample
                            collector.add_sample(densities)
                            
                            # Get predicted angle for display
                            angle, _ = detector.calculate_steering_angle(edges)
                            
                            # Update display
                            print_status(collector, current_human_angle, angle, densities, best_result)
                            
                            last_sample_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        detector.cleanup()
        print("\nCamera cleaned up.")
        
        # Show final summary
        if best_result:
            print(f"\nFinal best weights: {best_result['weights']}")
            print(f"Final best MAE: {best_result['mae']:.2f}°")
            print("\nTo apply these weights, run with 'S' to save, or manually update config.py:")
            print(f"REGION_WEIGHTS = [{best_result['weights'][0]:.3f}, {best_result['weights'][1]:.3f}, {best_result['weights'][2]:.3f}, {best_result['weights'][3]:.3f}, {best_result['weights'][4]:.3f}]")


if __name__ == "__main__":
    main()

