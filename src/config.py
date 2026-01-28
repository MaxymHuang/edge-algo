"""Configuration parameters for edge detection and camera settings."""

# Canny edge detection thresholds
# Lower values = more sensitive (detect more edges)
# Higher values = less sensitive (detect fewer, stronger edges)
CANNY_LOW_THRESHOLD = 20   # Lowered for better sparse edge detection
CANNY_HIGH_THRESHOLD = 100  # Lowered for better edge detection

# Camera settings
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 30
CAMERA_INDEX = 0  # Camera index (0 for Cam0, 1 for Cam1, etc.)

# Steering determination thresholds
EDGE_DENSITY_THRESHOLD = 0.1  # 10% difference to determine turn direction
MIN_EDGE_DENSITY = 0.005  # Minimum edge density to consider valid (lowered for sparse scenarios)

# Path-following mode configuration
PATH_FOLLOWING_MODE = True  # Enable path-following (steer towards least dense region)
MIN_PATH_DENSITY_THRESHOLD = 0.02  # Maximum density threshold for a region to be considered a valid path
PATH_STEERING_THRESHOLD = 0.005  # Minimum density difference between regions to trigger steering
SPARSE_EDGE_MODE = True  # Enable enhanced sensitivity for sparse scenarios

# Steering angle configuration
# Discrete steering angle classes in degrees
STEERING_ANGLE_CLASSES = [-45, -30, -15, 0, 15, 30, 45]

# Density weighting factors for each region (higher weight = more influence)
# Regions: [far-left, left, center, right, far-right]
# Outer regions weighted more heavily as lane edges typically appear there
REGION_WEIGHTS = [1.5, 1.2, 0.8, 1.2, 1.5]

# Thresholds for angle classification
# Minimum density difference between left and right regions to trigger steering
ANGLE_DENSITY_THRESHOLD = 0.05  # 5% difference minimum
# Score threshold for determining if steering is needed (lowered for path-following)
STEERING_SCORE_THRESHOLD = 0.05  # Lowered for better sparse scenario handling

# ML Model Configuration
LANE_DETECTION_METHOD = "edge"  # Options: "edge", "fast_scnn", "yolo", "hybrid"
ML_MODEL_CONFIDENCE_THRESHOLD = 0.5
HYBRID_ML_WEIGHT = 0.7  # Weight for ML predictions in hybrid mode
HYBRID_EDGE_WEIGHT = 0.3  # Weight for edge detection in hybrid mode

# Fast SCNN Settings
FAST_SCNN_MODEL_PATH = "models/fast_scnn/fast_scnn_quantized.onnx"  # Use quantized ONNX model
FAST_SCNN_INPUT_SIZE = (320, 240)  # Reduced resolution for Pi 5 performance
FAST_SCNN_FRAME_SKIP = 3  # Process every Nth frame (default: every 3rd)

# YOLOv8 Settings
YOLO_MODEL_PATH = "models/yolo/yolov8n_lane_seg_quantized.onnx"  # Use Nano model, quantized
YOLO_INPUT_SIZE = (416, 416)  # Standard YOLO input size (reduced from 640)
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_FRAME_SKIP = 5  # Process every Nth frame (YOLO is slower)

