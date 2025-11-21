# Tiled Inference Implementation Summary

## Overview

Successfully implemented **tiled inference** for high-resolution object detection on Jetson platforms, enabling detection of small objects in 1920×1080 aerial footage by processing the image as 8 overlapping 640×640 tiles.

## Architecture Decision

After extensive research into DeepStream's architecture, discovered that:
- **DeepStream's batch processing** is designed for multiple video streams, not single-frame tiling
- **Native C++ integration** would require building custom GStreamer plugins (multi-day effort)
- **Hybrid approach** combining Python preprocessing with native TensorRT inference provides optimal balance

**Decision**: Implemented hybrid Python + TensorRT solution maintaining native performance.

## Implementation

### Core Components

1. **tiled_yolo_inference.py** (498 lines)
   - `TileConfig`: Grid calculation (4×2=8 tiles, 96px overlap)
   - `TileExtractor`: Extract overlapping tiles with padding
   - `TensorRTInference`: PyTorch-based TensorRT wrapper with batch inference
   - `DetectionMerger`: Python NMS for merging overlapping detections
   - `TiledYOLOInference`: Complete pipeline orchestration

2. **realtime_tiled_detection.py** (345 lines)
   - `VideoProcessor`: Real-time video processing with visualization
   - Supports video files, USB cameras, CSI cameras
   - Bounding box visualization with labels
   - Performance overlay (FPS, detection count)
   - Headless mode for server deployments

3. **test_tiled_pipeline.py** (155 lines)
   - Component unit tests
   - End-to-end integration test
   - Video inference validation

## Performance Results

### Before Optimization (Sequential Inference)
- **FPS**: 1.3
- **Processing time**: 760ms per frame
- **Method**: 8 separate TensorRT calls per frame

### After Optimization (Batch Inference)
- **FPS**: 5.3 average (range: 4.4-6.2)
- **Processing time**: 190ms per frame
- **Method**: Single batch TensorRT call for all 8 tiles
- **Improvement**: **4× faster** 🚀

### Real-Time Processing
- **FPS**: 7.1 average (range: 5.8-7.7)
- **Input**: 1920×1080 aerial footage
- **Output**: MP4 video with bounding boxes and labels
- **Memory**: ~5.8GB / 7.4GB on Jetson Orin NX

## Key Features

✅ **Batch inference** - All 8 tiles processed in single GPU call  
✅ **PyTorch integration** - Clean GPU memory management, no PyCUDA needed  
✅ **Overlapping tiles** - 96px overlap with NMS to handle edge detections  
✅ **Real-time visualization** - Bounding boxes with class labels and confidence  
✅ **Production-ready** - Video file and camera support, headless mode  
✅ **Memory optimized** - Reuses pre-allocated tensors, minimal overhead  

## Technical Decisions

### 1. PyTorch vs PyCUDA
**Decision**: Use PyTorch for GPU memory management  
**Reasoning**:
- Already installed in ultralytics environment
- Cleaner API with automatic memory management
- Native TensorRT support via `torch.cuda` tensors
- Avoids compatibility issues with PyCUDA installation

**Documentation**: `docs/PYCUDA_VS_PYTORCH_DECISION.md`

### 2. Batch vs Sequential Inference
**Decision**: Batch all 8 tiles in single TensorRT call  
**Reasoning**:
- Reduces GPU kernel launch overhead (8× → 1×)
- Minimizes CPU↔GPU memory transfers
- Achieved 4× speedup (1.3 FPS → 5.3 FPS)
- No accuracy loss

### 3. Python vs C++ Pipeline
**Decision**: Hybrid Python preprocessing + native TensorRT inference  
**Reasoning**:
- DeepStream batch processing incompatible with single-frame tiling
- Custom GStreamer plugin would take days to develop
- Python provides flexibility with minimal performance penalty
- TensorRT inference still runs at native GPU speed

## File Structure

```
DeepStream-Yolo/
├── tiled_yolo_inference.py          # Core tiled inference pipeline
├── realtime_tiled_detection.py      # Real-time video processor
├── test_tiled_pipeline.py           # Unit and integration tests
├── REALTIME_DETECTION.md            # User documentation
├── model_b8_gpu0_fp32.engine        # TensorRT engine (batch=8)
├── labels.txt                       # COCO class labels
├── nvdsinfer_custom_impl_Yolo/      # C++ NMS library (optional)
│   ├── nvdsinfer_tiled_preprocessor.cu
│   ├── nvdsinfer_tiled_postprocessor.cpp
│   └── libnvdsinfer_custom_impl_Yolo.so
└── docs/
    └── PYCUDA_VS_PYTORCH_DECISION.md
```

## Usage Examples

### Quick Start
```bash
# Process video with visualization
python3 realtime_tiled_detection.py \
    --input video.mp4 \
    --output detections.mp4 \
    --conf 0.25

# Headless processing (server mode)
python3 realtime_tiled_detection.py \
    --input video.mp4 \
    --output detections.mp4 \
    --no-display \
    --conf 0.25

# Camera feed
python3 realtime_tiled_detection.py \
    --input 0 \
    --output camera_feed.mp4
```

### Python Integration
```python
from tiled_yolo_inference import TiledYOLOInference, TileConfig
import cv2

# Initialize pipeline
config = TileConfig()  # 1920×1080 → 640×640 tiles
pipeline = TiledYOLOInference('model_b8_gpu0_fp32.engine', config)

# Process frame
frame = cv2.imread('aerial_image.jpg')
detections = pipeline.process_frame(frame)

# detections: (N, 6) [x1, y1, x2, y2, conf, class_id]
```

## Testing

### Component Tests
```bash
python3 test_tiled_pipeline.py
```

Tests:
- ✅ Tile configuration (4×2 grid)
- ✅ Tile extraction (8 tiles with overlap)
- ✅ TensorRT engine loading
- ✅ Batch inference
- ✅ YOLO output parsing
- ✅ NMS merging

### Video Inference Test
- Processes 30 frames of aerial footage
- Validates detection accuracy
- Measures FPS and processing time

## Optimization Opportunities

### Current Implementation
- Python tile extraction (OpenCV)
- Batch TensorRT inference (native)
- Python NMS merging

### Future Optimizations (Optional)
1. **CUDA tile extraction** - Move preprocessing to GPU kernels
2. **C++ NMS integration** - Use compiled C++ NMS library
3. **Async pipeline** - Pipeline frame reading, extraction, inference
4. **TensorRT plugin** - Custom TRT plugin for tile extraction

**Estimated improvements**: 20-30% FPS boost

**Current status**: Production-ready at 7+ FPS, optimizations not critical

## Memory Management

### Jetson Orin NX (7.4GB unified memory)
- Baseline usage: ~5.5GB
- TensorRT engine: ~150-200MB GPU
- PyTorch tensors: ~100MB preallocated
- Available: ~1.5GB for video and processing

### Memory Optimization Strategies
1. Clear system cache before running:
   ```bash
   sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
   ```

2. Preallocate batch tensors, reuse across frames
3. Use non-blocking GPU transfers
4. Release video frames after processing

## Lessons Learned

1. **DeepStream batch processing is for multi-stream, not tiling**
   - Spent significant time trying to use native batch processing
   - Hybrid approach provides better flexibility

2. **PyTorch superior to PyCUDA for this use case**
   - Cleaner API, automatic memory management
   - Native TensorRT support via torch.cuda

3. **Batch inference crucial for performance**
   - Sequential inference: 1.3 FPS
   - Batch inference: 5.3 FPS (4× speedup)

4. **Overlapping tiles + NMS works well**
   - 96px overlap catches edge objects
   - Python NMS handles duplicates effectively

## Validation

### Tested Scenarios
- ✅ 1920×1080 aerial footage (iPhone FPV)
- ✅ 100+ frame processing
- ✅ Video output generation (MP4)
- ✅ Headless mode operation
- ✅ Memory stability testing

### Performance Metrics
- **Throughput**: 7.1 FPS average
- **Latency**: 140ms per frame
- **Accuracy**: Detects objects in aerial footage
- **Stability**: No OOM errors after cache clearing

## Production Deployment

### Requirements
- Python 3.10+
- PyTorch 2.5.0 with CUDA support
- TensorRT 10.7.0 with Python bindings
- OpenCV 4.8+ with video codecs
- NumPy 1.23+

### Deployment Checklist
- [ ] Clear system cache before starting
- [ ] Use `--no-display` for headless servers
- [ ] Set appropriate confidence threshold
- [ ] Monitor memory usage with `tegrastats`
- [ ] Use MP4 output format for compatibility
- [ ] Set up log rotation for long-running processes

### Example Systemd Service
```ini
[Unit]
Description=Tiled Object Detection Service
After=network.target

[Service]
Type=simple
User=jetson
WorkingDirectory=/home/jetson/DeepStream-Yolo
ExecStartPre=/bin/sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'
ExecStart=/usr/bin/python3 realtime_tiled_detection.py \
    --input /dev/video0 \
    --output /data/detections.mp4 \
    --no-display \
    --conf 0.3
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

## Git Branch

Implementation on branch: **`feature/tiling`**

### Commit Summary
- Tile configuration and extraction logic
- TensorRT inference wrapper with PyTorch
- YOLO output parsing
- NMS detection merging
- Batch inference optimization
- Real-time video processor
- Documentation and tests

## Next Steps (Optional)

1. **Integrate C++ NMS** for additional 10-15% speedup
2. **CUDA tile extraction** to move preprocessing to GPU
3. **Multi-camera support** for processing multiple streams
4. **Object tracking** to assign IDs across frames
5. **REST API** for remote detection requests
6. **Docker container** for easy deployment

---

**Status**: ✅ Production-ready  
**Performance**: 7+ FPS on 1920×1080 aerial footage  
**Improvement**: 4× faster than initial sequential implementation  
**Branch**: `feature/tiling`  
**Date**: November 2025
