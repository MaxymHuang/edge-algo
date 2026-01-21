# Vision Algorithm - Edge Detection Steering System

Edge detection system for Raspberry Pi 5 that analyzes camera input to determine steering direction (left/right) using OpenCV.

## Features

- Real-time edge detection using Canny algorithm
- Steering direction analysis based on edge position
- Web-based console with:
  - Live camera feed
  - Edge detection visualization
  - Real-time log messages
  - Steering direction indicator

## Requirements

- Raspberry Pi 5
- Raspberry Pi Camera Module
- Python 3.10+
- Node.js 18+ (for frontend)

## Installation

### 1. Install Python Dependencies

Using `uv` (recommended):

```bash
uv pip install -e .
```

Or using pip:

```bash
pip install -e .
```

### 2. Build Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

## Usage

### Run Web Server

```bash
python -m src.run_server
```

Or using uvicorn directly:

```bash
uvicorn src.web_server:app --host 0.0.0.0 --port 8000
```

### Access Web Console

Open your browser and navigate to:

```
http://localhost:8000
```

Or from another device on the same network:

```
http://<raspberry-pi-ip>:8000
```

### Run Standalone (without web interface)

```bash
python -m src.main
```

## Configuration

Edit `src/config.py` to adjust:

- Canny edge detection thresholds
- Camera resolution and framerate
- Steering determination thresholds

## Project Structure

```
vision-algo/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── edge_detection.py      # Core edge detection logic
│   ├── main.py                # Standalone application
│   ├── web_server.py          # FastAPI web server
│   └── run_server.py          # Server entry point
├── frontend/                  # React/TypeScript frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── CameraFeed.tsx
│   │   │   ├── EdgeDetection.tsx
│   │   │   └── LogMessages.tsx
│   │   └── main.tsx
│   └── package.json
├── static/                    # Built frontend (generated)
├── pyproject.toml
└── README.md
```

## Algorithm Details

### Steering Angle Determination Algorithm

The system uses a **five-region weighted density analysis** to determine steering angles. The algorithm processes camera frames through the following steps:

#### 1. Edge Detection
- Converts the camera frame to grayscale
- Applies Gaussian blur to reduce noise
- Uses **Canny edge detection** with adaptive thresholds based on image median
- Produces a binary edge map where edge pixels are white (255) and non-edge pixels are black (0)

#### 2. Five-Region Division
The frame is divided horizontally into five equal-width regions:
- **Far-Left**: 0-20% of frame width
- **Left**: 20-40% of frame width  
- **Center**: 40-60% of frame width
- **Right**: 60-80% of frame width
- **Far-Right**: 80-100% of frame width

#### 3. Edge Density Calculation
For each region, the algorithm calculates **edge density** as the proportion of edge pixels:
```
density = (number of edge pixels) / (total pixels in region)
```

This produces five density values: `d_far_left`, `d_left`, `d_center`, `d_right`, `d_far_right`

#### 4. Weighted Density Application
Each region density is multiplied by a weight factor to emphasize outer regions (where lane edges typically appear):
- Far-Left: weight = 1.5
- Left: weight = 1.2
- Center: weight = 0.8
- Right: weight = 1.2
- Far-Right: weight = 1.5

```
weighted_density[i] = density[i] × weight[i]
```

#### 5. Side Score Calculation
The weighted densities are combined into left and right scores:
- **Left Score** = weighted_density[far_left] + weighted_density[left]
- **Right Score** = weighted_density[right] + weighted_density[far_right]
- The center region is excluded from scoring as it represents the current path

#### 6. Steering Score Normalization
A normalized steering score is calculated:
```
total_score = left_score + right_score
steering_score = (right_score - left_score) / total_score
```

The steering score ranges from:
- **-1.0**: Hard left (all edges on left side)
- **0.0**: Straight (balanced edges)
- **+1.0**: Hard right (all edges on right side)

#### 7. Validation Checks
Before determining an angle, the algorithm performs validation:
1. **Minimum Density Check**: If average edge density < `MIN_EDGE_DENSITY` (0.01), return 0° (straight)
2. **Score Threshold Check**: If `|steering_score| < STEERING_SCORE_THRESHOLD` (0.1), return 0° (straight)

#### 8. Angle Classification
If validation passes, the steering score is mapped to discrete angle classes:
- Available angles: `[-45°, -30°, -15°, 0°, 15°, 30°, 45°]`
- The algorithm finds the closest angle class using:
  ```
  target_angle = steering_score × max_angle (45°)
  selected_angle = closest_angle_class_to(target_angle)
  ```

#### Algorithm Flow Diagram

```
Camera Frame
    ↓
Grayscale Conversion + Gaussian Blur
    ↓
Canny Edge Detection
    ↓
Five-Region Division (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
    ↓
Calculate Edge Density per Region
    ↓
Apply Region Weights [1.5, 1.2, 0.8, 1.2, 1.5]
    ↓
Calculate Left Score (far-left + left)
Calculate Right Score (right + far-right)
    ↓
Normalize: steering_score = (right - left) / (right + left)
    ↓
Validation Checks
    ↓
Map to Discrete Angle Class [-45°, -30°, -15°, 0°, 15°, 30°, 45°]
    ↓
Output Steering Angle
```

#### Example Calculation

Given edge densities:
- Far-Left: 0.15 (15%)
- Left: 0.10 (10%)
- Center: 0.05 (5%)
- Right: 0.08 (8%)
- Far-Right: 0.12 (12%)

**Step 1**: Apply weights
- Far-Left: 0.15 × 1.5 = 0.225
- Left: 0.10 × 1.2 = 0.120
- Center: 0.05 × 0.8 = 0.040
- Right: 0.08 × 1.2 = 0.096
- Far-Right: 0.12 × 1.5 = 0.180

**Step 2**: Calculate side scores
- Left Score: 0.225 + 0.120 = 0.345
- Right Score: 0.096 + 0.180 = 0.276

**Step 3**: Calculate steering score
- Total Score: 0.345 + 0.276 = 0.621
- Steering Score: (0.276 - 0.345) / 0.621 = -0.111

**Step 4**: Map to angle
- Steering score = -0.111 (slight left bias)
- Target angle = -0.111 × 45° = -5.0°
- Closest angle class: **-15°** (left turn)

#### Configuration Parameters

Key parameters in `src/config.py`:
- `STEERING_ANGLE_CLASSES`: Available steering angles `[-45, -30, -15, 0, 15, 30, 45]`
- `REGION_WEIGHTS`: Weight factors `[1.5, 1.2, 0.8, 1.2, 1.5]`
- `MIN_EDGE_DENSITY`: Minimum density threshold `0.01` (1%)
- `STEERING_SCORE_THRESHOLD`: Minimum score difference `0.1` (10%)

## License

This project is provided as-is for educational and development purposes.

