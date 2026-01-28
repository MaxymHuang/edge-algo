# Vision Algorithm - Edge Detection Steering System

Edge detection system for Raspberry Pi 5 that analyzes camera input to determine steering direction (left/right) using OpenCV.

## Features

- Real-time edge detection using Canny algorithm
- **ML-based lane detection** using Fast SCNN and YOLOv8 segmentation models
- **Hybrid detection mode** combining ML predictions with edge detection
- Steering direction analysis based on lane position
- Web-based console with:
  - Live camera feed
  - Edge detection visualization
  - ML lane detection visualization
  - Real-time log messages
  - Steering direction indicator
  - Model selection interface

## Requirements

- Raspberry Pi 5 (8GB RAM recommended for ML models)
- Raspberry Pi Camera Module
- Python 3.10+
- Node.js 18+ (for frontend)
- **64-bit OS** (required for ML models - 32-bit limits processes to 3GB)

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

### Run Weight Tuner (Web Interface)

The weight tuner provides a web-based interface for collecting training data and optimizing region weights using Bayesian optimization.

```bash
# First, build the frontend (if not already built)
cd frontend
npm install
npm run build
cd ..

# Run the tuner server
uv run python -m src.run_tuner_server
```

Or using uvicorn directly:

```bash
uvicorn src.tuner_server:app --host 0.0.0.0 --port 8001
```

Then open your browser and navigate to:

```
http://localhost:8001
```

**Tuner Features:**
- Real-time camera feed and edge detection visualization
- Interactive steering angle controls (buttons for all angle classes)
- Live sample collection with visual feedback
- One-click Bayesian optimization
- View optimization results (best weights and MAE)
- Save optimized weights to config
- Reset samples for new training sessions

**Usage:**
1. Use the steering controls to set your desired steering angle while driving
2. The system automatically collects samples (edge densities + human angle)
3. Once you have at least 10 samples, click "Run Optimization"
4. Review the results and click "Save Weights" to apply them to the config

## ML Model Setup

The system supports ML-based lane detection using Fast SCNN and YOLOv8 segmentation models.

### Download and Prepare Models

1. **Download models**:
   ```bash
   python scripts/download_models.py
   ```

2. **Convert to ONNX** (if needed):
   - Fast SCNN: Download from GitHub repositories (antoniojkim/Fast-SCNN)
   - YOLOv8: Automatically downloaded via ultralytics

3. **Quantize models** (recommended for Pi 5):
   ```bash
   # Quantize Fast SCNN
   python scripts/quantize_model.py models/fast_scnn/fast_scnn.onnx models/fast_scnn/fast_scnn_quantized.onnx
   
   # Quantize YOLOv8
   python scripts/quantize_model.py models/yolo/yolov8n_lane_seg.onnx models/yolo/yolov8n_lane_seg_quantized.onnx
   ```

### Model Performance

- **Fast SCNN**: ~2-5 FPS, ~150-300 MB RAM
- **YOLOv8n**: ~1-2 FPS, ~200-400 MB RAM
- **Edge Detection**: ~30 FPS, ~10-20 MB RAM

**Note**: ML models use frame skipping (process every 3rd-5th frame) to maintain responsiveness. See `RESOURCE_EVALUATION.md` for detailed analysis.

### Detection Methods

Configure in `src/config.py` or via web UI:

- **`edge`**: Traditional Canny edge detection (fastest, 30 FPS)
- **`fast_scnn`**: Fast SCNN semantic segmentation (balanced, 2-5 FPS)
- **`yolo`**: YOLOv8 segmentation (most accurate, 1-2 FPS)
- **`hybrid`**: Combines ML predictions (70%) with edge detection (30%)

## Configuration

Edit `src/config.py` to adjust:

- Canny edge detection thresholds
- Camera resolution and framerate
- Steering determination thresholds
- **ML model settings** (paths, input sizes, frame skipping)

## Project Structure

```
vision-algo/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters
│   ├── edge_detection.py      # Core edge detection logic
│   ├── main.py                # Standalone application
│   ├── web_server.py          # FastAPI web server
│   ├── tuner_server.py        # Weight tuner web server
│   ├── weight_tuner.py        # Bayesian optimization logic
│   ├── run_server.py          # Server entry point
│   ├── run_tuner.py           # Terminal-based tuner (legacy)
│   └── run_tuner_server.py     # Tuner server entry point
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

## Bayesian Optimization for Weight Tuning

### What is Bayesian Optimization?

Bayesian optimization is a powerful technique for finding the optimal values of expensive-to-evaluate functions. Unlike grid search or random search, it uses a probabilistic model (surrogate model) to intelligently explore the parameter space, balancing **exploration** (trying new areas) and **exploitation** (refining promising areas).

#### Key Concepts

1. **Surrogate Model**: A probabilistic model (typically a Gaussian Process) that approximates the objective function based on observed data points
2. **Acquisition Function**: A strategy for selecting the next point to evaluate, balancing exploration vs exploitation
3. **Prior Knowledge**: Incorporates beliefs about the function before seeing data
4. **Posterior Updates**: Updates beliefs after each evaluation using Bayes' theorem

#### How Bayesian Optimization Works

```
1. Initialize with a few random samples
   ↓
2. Build surrogate model (probabilistic approximation of objective function)
   ↓
3. Use acquisition function to select next promising point
   ↓
4. Evaluate objective function at selected point
   ↓
5. Update surrogate model with new observation
   ↓
6. Repeat steps 3-5 until convergence or budget exhausted
```

**Advantages over brute-force methods:**
- **Efficient**: Requires fewer function evaluations
- **Adaptive**: Learns from previous evaluations
- **Handles noise**: Works well with noisy objective functions
- **Global optimization**: Can escape local minima

### Application in This Project

In this vision steering system, Bayesian optimization is used to automatically tune the **region weights** (`REGION_WEIGHTS`) that determine how much influence each horizontal region of the frame has on steering angle prediction.

#### Problem Statement

The steering algorithm uses five region weights: `[far_left, left, center, right, far_right]`. These weights control how edge densities in each region contribute to the final steering decision. Finding optimal weights manually is difficult because:

- The relationship between weights and steering accuracy is complex and non-linear
- Each evaluation requires collecting real-world driving data
- The search space is continuous (infinite possible weight combinations)
- Weights must be symmetric (left/right symmetry constraint)

#### Implementation Details

The optimization is implemented in `src/weight_tuner.py` using **Optuna** with the **TPE (Tree-structured Parzen Estimator)** sampler.

##### 1. Objective Function

The optimization minimizes **Mean Absolute Error (MAE)** between predicted and human-provided steering angles:

```python
MAE = mean(|predicted_angle - human_angle|)
```

For each trial:
1. Sample candidate weights from the search space
2. Apply weights to all collected training samples
3. Calculate predicted angles using the candidate weights
4. Compute MAE across all samples
5. Return MAE as the objective value

##### 2. Search Space and Constraints

The optimization uses **symmetric weight constraints** to reduce the search space:

- **Outer weights** (far-left, far-right): Range `[1.0, 2.0]` - shared value
- **Inner weights** (left, right): Range `[0.8, 1.5]` - shared value  
- **Center weight**: Range `[0.5, 1.2]` - independent value

Final weight vector: `[outer, inner, center, inner, outer]`

This constraint:
- Reduces search space from 5D to 3D
- Enforces left-right symmetry (realistic for lane detection)
- Speeds up convergence

##### 3. TPE Sampler

The project uses **TPE (Tree-structured Parzen Estimator)** as the Bayesian optimization algorithm:

- **How it works**: Models the distribution of good vs bad hyperparameters
  - Splits observations into "good" (top quantile) and "bad" (bottom quantile) groups
  - Models each group using Parzen estimators (kernel density estimation)
  - Samples new candidates from the "good" distribution, avoiding the "bad" distribution

- **Why TPE**: 
  - Handles mixed continuous/discrete spaces well
  - Efficient for moderate number of trials (30-100)
  - Built into Optuna with good defaults

##### 4. Data Collection Workflow

The optimization process follows this workflow:

```
1. Collect Training Data
   ├─ Run camera feed in real-time
   ├─ Human provides steering input via keyboard (arrow keys)
   ├─ System captures: (edge_densities, human_angle) pairs
   └─ Store samples in buffer (SteeringDataCollector)

2. Run Optimization
   ├─ Initialize Optuna study with TPE sampler
   ├─ For each trial (default: 30 trials):
   │  ├─ Sample candidate weights
   │  ├─ Evaluate MAE on all collected samples
   │  └─ Update TPE model
   └─ Return best weights and MAE

3. Apply Results
   ├─ Save best weights to config.py
   └─ Use optimized weights in production
```

#### Code Structure

The implementation consists of three main components:

1. **`SteeringDataCollector`**: Collects and buffers training samples
   - Stores `(densities, human_angle)` pairs
   - Maintains a rolling buffer (default: 1000 samples)
   - Provides sample access for optimization

2. **`WeightOptimizer`**: Performs Bayesian optimization
   - Wraps Optuna study creation and execution
   - Implements symmetric weight constraints
   - Manages optimization trials and results

3. **`calculate_steering_with_weights()`**: Evaluates candidate weights
   - Takes densities and weights as input
   - Computes predicted steering angle
   - Used by optimizer to evaluate each trial

#### Usage

To use the weight tuner:

```bash
uv run python -m src.run_tuner
```

**Interactive Controls:**
- **Arrow Keys**: Provide human steering input (left/right/straight)
- **O**: Run optimization with current samples
- **S**: Save best weights to `config.py`
- **R**: Reset/clear collected samples
- **Q**: Quit

**Best Practices:**
1. Collect diverse samples: drive in various scenarios (straight, turns, curves)
2. Collect sufficient data: at least 100-500 samples for reliable optimization
3. Run multiple optimizations: verify consistency across runs
4. Validate on new data: test optimized weights on unseen scenarios

#### Optimization Parameters

Key parameters in `WeightOptimizer`:

- **`n_trials`**: Number of optimization trials (default: 30)
  - More trials = better optimization but slower
  - Recommended: 30-100 trials depending on data size

- **Weight ranges**: Define search space bounds
  - `outer_weight_range`: `(1.0, 2.0)` - outer region emphasis
  - `inner_weight_range`: `(0.8, 1.5)` - inner region emphasis
  - `center_weight_range`: `(0.5, 1.2)` - center region emphasis

#### Example Optimization Result

After optimization, you might see:

```
Optimization complete. Best MAE: 8.5°
Best weights: outer=1.65, inner=1.15, center=0.92
Final weight vector: [1.65, 1.15, 0.92, 1.15, 1.65]
```

This means:
- The optimized weights reduce prediction error to 8.5° average
- Outer regions are weighted 1.65× (emphasizing lane edges)
- Inner regions are weighted 1.15× (moderate emphasis)
- Center region is weighted 0.92× (slight de-emphasis)

#### Why Bayesian Optimization is Ideal Here

1. **Expensive Evaluation**: Each weight evaluation requires processing all training samples
2. **Noisy Objective**: Steering prediction has inherent variability
3. **Limited Budget**: Can only run a limited number of trials
4. **Non-convex**: The relationship between weights and accuracy is complex
5. **Continuous Space**: Weights are continuous values, not discrete choices

Traditional methods (grid search, random search) would require:
- Grid search: 10^5 = 100,000 evaluations (for 5 weights with 10 values each)
- Random search: Thousands of random trials
- Bayesian optimization: ~30-100 intelligent trials

## License

This project is provided as-is for educational and development purposes.

