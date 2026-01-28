# Resource Evaluation: ML Models on Raspberry Pi 5 (8GB RAM)

## Executive Summary

**Verdict: FEASIBLE with optimizations, but with performance trade-offs**

The ML model integration is **technically feasible** on Raspberry Pi 5 with 8GB RAM, but will require:
- Model quantization and optimization
- Reduced frame processing rates (1-5 FPS for ML models vs 30 FPS for edge detection)
- Careful memory management
- Use of smaller model variants (YOLOv8n, optimized Fast SCNN)

## Hardware Specifications

### Raspberry Pi 5 8GB Capabilities
- **CPU**: Broadcom BCM2712, 64-bit quad-core Arm Cortex-A76 @ 2.4GHz
- **GPU**: VideoCore VII @ 800MHz (OpenGL ES 3.1, Vulkan 1.2)
- **RAM**: 8GB LPDDR4X
- **Performance**: 2-3× CPU performance vs Pi 4, 2-2.5× faster GPU
- **AI Inference**: ~12 FPS for YOLOv8n at 640×640 (with ncnn framework)

## Memory Requirements Analysis

### Current System Baseline
- **OS + System**: ~1-2GB
- **Python Runtime**: ~100-200MB
- **OpenCV + NumPy**: ~50-100MB
- **FastAPI + Web Server**: ~50-100MB
- **Camera Buffers**: ~10-50MB (640×480 RGB frames)
- **Edge Detection**: ~10-20MB (minimal)
- **Total Baseline**: ~1.5-2.5GB

### ML Model Memory Footprint

#### YOLOv8 Segmentation Models
| Model | Model Size | RAM Usage | Inference Time (Pi 5) | FPS Estimate |
|-------|------------|-----------|----------------------|--------------|
| YOLOv8n | 6.2 MB | ~200-400 MB | ~1000ms | ~1 FPS |
| YOLOv8s | 21.5 MB | ~400-600 MB | ~2700ms | ~0.4 FPS |
| YOLOv8m | 41.7 MB | ~600-800 MB | ~10000ms | ~0.1 FPS |

**PyTorch Runtime Overhead**: ~300-500MB (first model load)
- Additional models: +100-200MB each
- Model weights in memory: ~2-4× model file size

#### Fast SCNN
- **Model Size**: ~5-10 MB (estimated)
- **RAM Usage**: ~150-300 MB (estimated)
- **Inference Time**: Unknown on Pi 5 (designed for real-time, but likely 200-500ms)
- **FPS Estimate**: ~2-5 FPS (optimistic)

### Total Memory Usage Scenarios

#### Scenario 1: Single Model (YOLOv8n)
- Baseline: 1.5-2.5 GB
- PyTorch: 300-500 MB
- YOLOv8n: 200-400 MB
- Frame buffers: 20-50 MB
- **Total: ~2.0-3.5 GB** ✅ **FEASIBLE**

#### Scenario 2: Single Model (Fast SCNN)
- Baseline: 1.5-2.5 GB
- PyTorch: 300-500 MB
- Fast SCNN: 150-300 MB
- Frame buffers: 20-50 MB
- **Total: ~2.0-3.5 GB** ✅ **FEASIBLE**

#### Scenario 3: Both Models Loaded (Hybrid Mode)
- Baseline: 1.5-2.5 GB
- PyTorch: 300-500 MB
- YOLOv8n: 200-400 MB
- Fast SCNN: 150-300 MB
- Frame buffers: 20-50 MB
- **Total: ~2.2-4.0 GB** ⚠️ **TIGHT BUT FEASIBLE**

#### Scenario 4: With Edge Detection + Web Server
- All above + edge detection: +10-20 MB
- Web server overhead: +50-100 MB
- **Total: ~2.3-4.1 GB** ⚠️ **TIGHT BUT FEASIBLE**

## Performance Considerations

### Inference Speed Constraints

**Current System**: Edge detection at 30 FPS (640×480)

**With ML Models**:
- **YOLOv8n**: ~1 FPS (1000ms per frame) - **30× slower**
- **Fast SCNN**: ~2-5 FPS (200-500ms per frame) - **6-15× slower**

### Real-World Impact

For RC car steering:
- **Edge Detection**: Can process every frame (30 FPS)
- **ML Models**: Must skip frames or process every Nth frame
- **Recommended**: Process every 3rd-5th frame with ML, use edge detection for intermediate frames

### Optimization Strategies

1. **Model Quantization (INT8)**
   - Reduces model size by ~4×
   - Reduces memory by ~3-4×
   - Speeds up inference by ~2-3×
   - **Trade-off**: ~1-3% accuracy loss

2. **ONNX Runtime**
   - Faster inference than PyTorch (20-30% improvement)
   - Lower memory overhead
   - Better CPU optimization

3. **Frame Skipping**
   - Process every Nth frame with ML
   - Use edge detection for intermediate frames
   - Hybrid approach maintains responsiveness

4. **Input Resolution Reduction**
   - Current: 640×480
   - ML Input: 320×240 or 416×416 (YOLO standard)
   - Reduces processing time by ~4×
   - **Trade-off**: Lower accuracy on small lanes

5. **Model Pruning**
   - Remove unnecessary layers
   - Reduce model complexity
   - Custom lightweight architecture

## Recommended Implementation Strategy

### Phase 1: Fast SCNN (Recommended First)
- **Why**: Designed for real-time embedded inference
- **Memory**: Lower footprint than YOLOv8
- **Speed**: Likely faster than YOLOv8n
- **Complexity**: Simpler integration (semantic segmentation)

**Configuration**:
- Model: Quantized Fast SCNN (INT8)
- Input: 320×240 or 416×416
- Processing: Every 3rd frame
- Fallback: Edge detection for skipped frames

### Phase 2: YOLOv8n (If Needed)
- **Why**: More accurate, multi-task capability
- **Memory**: Higher but manageable
- **Speed**: Slower (1 FPS)
- **Use Case**: When higher accuracy is critical

**Configuration**:
- Model: YOLOv8n quantized (INT8)
- Input: 416×416 (standard YOLO input)
- Processing: Every 5th frame
- Fallback: Edge detection or Fast SCNN

### Phase 3: Hybrid Mode
- **Why**: Best of both worlds
- **Memory**: Manageable with lazy loading
- **Speed**: Use faster model (Fast SCNN) for most frames
- **Accuracy**: Combine predictions for critical decisions

**Configuration**:
- Primary: Fast SCNN (every frame)
- Secondary: YOLOv8n (every 10th frame for validation)
- Fallback: Edge detection

## Critical Requirements

### 1. 64-bit Operating System
- **MUST**: Use 64-bit OS (Raspberry Pi OS 64-bit)
- **Why**: 32-bit OS limits processes to 3GB even with 8GB RAM
- **Impact**: 64-bit allows full 8GB access

### 2. Model Optimization
- **MUST**: Use quantized models (INT8)
- **MUST**: Convert to ONNX for faster inference
- **SHOULD**: Use smaller input resolutions
- **SHOULD**: Implement model lazy loading

### 3. Memory Management
- **MUST**: Load only one model at a time (unless hybrid mode)
- **SHOULD**: Unload models when switching
- **SHOULD**: Monitor memory usage
- **SHOULD**: Implement memory cleanup

### 4. Frame Processing Strategy
- **MUST**: Skip frames (process every Nth frame)
- **SHOULD**: Use edge detection for intermediate frames
- **SHOULD**: Cache ML predictions
- **SHOULD**: Lower camera framerate if needed (15-20 FPS)

## Performance Benchmarks (Estimated)

### YOLOv8n (Quantized, ONNX, 416×416)
- **Memory**: ~150-250 MB
- **Inference**: ~300-500ms
- **FPS**: ~2-3 FPS
- **Accuracy**: ~95% of FP32 model

### Fast SCNN (Quantized, ONNX, 320×240)
- **Memory**: ~100-200 MB
- **Inference**: ~100-200ms
- **FPS**: ~5-10 FPS
- **Accuracy**: ~97% of FP32 model

### Hybrid (Fast SCNN + Edge Detection)
- **Memory**: ~200-350 MB
- **Inference**: ~100-200ms (Fast SCNN) + ~10ms (edge)
- **FPS**: ~5-10 FPS (ML) + 30 FPS (edge)
- **Accuracy**: High (combines both methods)

## Recommendations

### ✅ DO
1. Start with Fast SCNN (quantized, ONNX)
2. Use 64-bit OS
3. Implement frame skipping (every 3rd frame)
4. Use smaller input resolution (320×240 or 416×416)
5. Monitor memory usage
6. Implement lazy model loading
7. Use edge detection as fallback

### ❌ DON'T
1. Don't use full-size models (YOLOv8m/l/x)
2. Don't process every frame with ML
3. Don't load multiple models simultaneously (unless necessary)
4. Don't use FP32 models (too slow, too much memory)
5. Don't use 32-bit OS
6. Don't use full camera resolution for ML input

## Conclusion

**The ML model integration is FEASIBLE on Raspberry Pi 5 with 8GB RAM**, but requires:

1. **Optimization**: Quantization, ONNX conversion, reduced input size
2. **Performance Trade-offs**: 1-5 FPS for ML vs 30 FPS for edge detection
3. **Smart Processing**: Frame skipping, hybrid approaches
4. **Memory Management**: Lazy loading, single model at a time

**Recommended Approach**:
- **Primary**: Fast SCNN (quantized, ONNX) with frame skipping
- **Fallback**: Edge detection for intermediate frames
- **Optional**: YOLOv8n for higher accuracy scenarios (with lower FPS)

The system will be **resource-intensive** but **manageable** with proper optimization. Expect **2-5 FPS** for ML-based lane detection vs **30 FPS** for edge detection, which is acceptable for RC car steering (steering decisions don't need 30 FPS).
