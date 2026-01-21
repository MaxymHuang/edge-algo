"""Configuration parameters for edge detection and camera settings."""

# Canny edge detection thresholds
# Lower values = more sensitive (detect more edges)
# Higher values = less sensitive (detect fewer, stronger edges)
CANNY_LOW_THRESHOLD = 30   # Lowered for better edge detection
CANNY_HIGH_THRESHOLD = 100  # Lowered for better edge detection

# Camera settings
CAMERA_RESOLUTION = (640, 480)
CAMERA_FRAMERATE = 30
CAMERA_INDEX = 0  # Camera index (0 for Cam0, 1 for Cam1, etc.)

# Steering determination thresholds
EDGE_DENSITY_THRESHOLD = 0.1  # 10% difference to determine turn direction
MIN_EDGE_DENSITY = 0.01  # Minimum edge density to consider valid

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
# Score threshold for determining if steering is needed
STEERING_SCORE_THRESHOLD = 0.1

