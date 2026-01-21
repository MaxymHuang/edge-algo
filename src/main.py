"""Main application entry point."""

import logging
import time
from .edge_detection import EdgeDetector
from .config import CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run edge detection and steering analysis."""
    detector = EdgeDetector()
    
    try:
        # Initialize camera
        detector.initialize_camera(CAMERA_RESOLUTION, CAMERA_FRAMERATE, CAMERA_INDEX)
        
        logger.info("Starting edge detection loop...")
        
        # Continuous loop
        while True:
            # Capture frame
            frame = detector.capture_frame()
            if frame is None:
                logger.warning("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Detect edges
            edges = detector.detect_edges(frame)
            if edges is None:
                logger.warning("Failed to detect edges")
                time.sleep(0.1)
                continue
            
            # Analyze and get steering direction and angle
            steering_data = detector.analyze_edge_position(edges)
            logger.info(f"Steering direction: {steering_data['direction']}, angle: {steering_data['angle']}°")
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()

