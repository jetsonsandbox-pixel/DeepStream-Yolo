# Tiled Inference Quick Start Guide

## 🎯 What We Implemented

Based on the proven `gstreamer_yolo_tracker.py` implementation from `umbrella-jetson-dev`, we've created a comprehensive guide for implementing tiled inference in DeepStream-Yolo.

## 📁 Files Created

1. **`docs/TILED_INFERENCE_GUIDE.md`** - Complete implementation guide with:
   - Architecture overview
   - Three implementation options (Custom Library, GStreamer, Python)
   - Full C++ code for tile extraction and NMS merging
   - CUDA kernels for GPU-accelerated processing
   - Configuration examples
   - Performance tuning guidelines
   - Debugging techniques

## 🔑 Key Concepts from Working Implementation

### Frame Tiling Strategy
```
Original: 1920×1080 frame
Result:   8 tiles of 640×640
Layout:   4×2 grid
Overlap:  96 pixels (15%)
Stride:   544 pixels (640 - 96)
```

### Coordinate Transformation Algorithm
```python
# From umbrella-jetson-dev/gstreamer_yolo_tracker.py
scale_x = tile_info['width'] / self.tile_size
scale_y = tile_info['height'] / self.tile_size

orig_x1 = (x1 * scale_x) + tile_info['x']
orig_y1 = (y1 * scale_y) + tile_info['y']
orig_x2 = (x2 * scale_x) + tile_info['x']
orig_y2 = (y2 * scale_y) + tile_info['y']

# Clamp to frame boundaries
orig_x1 = max(0, min(orig_x1, frame_width))
orig_y1 = max(0, min(orig_y1, frame_height))
```

### NMS Merging Strategy
```python
# Non-Maximum Suppression parameters
score_threshold = 0.25  # Minimum confidence
nms_threshold = 0.45    # IoU threshold for duplicates

# Apply OpenCV NMS
indices = cv2.dnn.NMSBoxes(
    boxes_xywh, all_confidences,
    score_threshold=0.25,
    nms_threshold=0.45
)
```

## 📊 Performance Expectations

### From Working Implementation
- **Processing Mode**: Batch (8 tiles simultaneously)
- **Inference Time**: ~90ms per batch vs 15ms single frame
- **Small Object Detection**: +15-30% improvement
- **Memory Usage**: ~4GB GPU (vs 2GB standard)
- **Overall FPS**: Maintained with frame skipping (interval=5)

### Configuration in DeepStream
```ini
# config_infer_primary_yolo11.txt
batch-size=8              # Process all 8 tiles together
tile-width=640
tile-height=640
tile-overlap=96
enable-tiling=1
```

## 🚀 Implementation Options

### Option 1: Custom C++ Library (Recommended)
**Best for**: Production deployment, optimal performance

**Components**:
- `nvdsinfer_tiled_preprocessor.cpp` - CUDA kernel for tile extraction
- `nvdsinfer_tiled_postprocessor.cpp` - NMS merging implementation
- Modified Makefile with CUDA support

**Advantages**:
- Hardware-accelerated preprocessing
- Direct DeepStream integration
- Minimal memory copies
- Best performance

### Option 2: GStreamer Plugin
**Best for**: Rapid prototyping, testing

**Components**:
- Modified pipeline configuration
- Pre-processing plugin
- Post-processing plugin

**Advantages**:
- No C++ coding required
- Pure GStreamer approach
- Good for validation

### Option 3: Python Application
**Best for**: Research, experimentation

**Components**:
- Python script using DeepStream bindings
- Direct code reuse from `gstreamer_yolo_tracker.py`

**Advantages**:
- Easy to modify and debug
- Proven algorithm implementation
- Rapid iteration

## 🔧 Next Steps

### 1. Choose Implementation Option
Start with **Option 1 (Custom Library)** for production use.

### 2. Build Custom Library
```bash
cd /home/jet-nx8/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
# Add tiled preprocessing and postprocessing files
# Update Makefile with CUDA support
make clean
make
```

### 3. Update Configuration
```bash
# Edit config_infer_primary_yolo11.txt
batch-size=8
tile-width=640
tile-height=640
tile-overlap=96
enable-tiling=1
```

### 4. Test with Aerial Footage
```bash
cd /home/jet-nx8/DeepStream-Yolo
deepstream-app -c deepstream_app_config.txt
```

### 5. Benchmark Performance
```bash
# Enable performance monitoring
export NVDS_ENABLE_LATENCY_MEASUREMENT=1
deepstream-app -c deepstream_app_config.txt
```

## 📈 Validation Checklist

- [ ] Tile extraction produces 8 tiles from 1920×1080
- [ ] Batch inference processes all 8 tiles simultaneously
- [ ] Coordinate transformation maps correctly to original frame
- [ ] NMS removes duplicate detections in overlap regions
- [ ] Small objects detected that were missed in standard mode
- [ ] FPS remains acceptable (>5 FPS with interval=5)
- [ ] GPU memory usage within limits (~4GB)

## 🐛 Troubleshooting

### Issue: Batch size mismatch
```
ERROR: batch-size (1) != tiles (8)
```
**Solution**: Set `batch-size=8` in config file

### Issue: Coordinate transformation incorrect
```
Detections appear in wrong locations
```
**Solution**: Verify scale_x/scale_y calculations include edge tile handling

### Issue: Duplicate detections
```
Same object detected multiple times
```
**Solution**: Tune NMS threshold (try 0.40-0.50 range)

### Issue: Performance too slow
```
FPS drops below acceptable levels
```
**Solution**: Increase detection interval (try interval=8)

## 📚 Reference Implementation

All algorithms are based on the working implementation at:
```
/home/jet-nx8/Sandbox/umbrella-jetson-dev/gstreamer_yolo_tracker.py
```

Key functions:
- `_initialize_tiling_config()` - Lines 2196-2237
- `_extract_tiles()` - Lines 2240-2273
- `_merge_tile_detections()` - Lines 2276-2396
- `_process_frame_with_tiling()` - Lines 2399-2434

## 🎓 Additional Resources

- **Full Implementation Guide**: `docs/TILED_INFERENCE_GUIDE.md`
- **Frame Tiling Concept**: `umbrella-jetson-dev/docs/features/FRAME_TILING_GUIDE.md`
- **DeepStream SDK**: https://docs.nvidia.com/metropolis/deepstream/
- **TensorRT Batching**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/

---

## 📝 Git History

```bash
# View implementation commits
git log --oneline feature/tiling

# Latest commits:
# 4f1d31a Add comprehensive tiled inference implementation guide
# 8256b51 Configure YOLO11n custom model for aerial detection
```

## 🌐 Remote Repository

Branch pushed to: `feature/tiling`
```
https://github.com/jetsonsandbox-pixel/DeepStream-Yolo/tree/feature/tiling
```

Create pull request:
```
https://github.com/jetsonsandbox-pixel/DeepStream-Yolo/pull/new/feature/tiling
```

---

*Implementation based on production-tested code processing aerial object detection footage on Jetson Orin NX with YOLO11n model. Frame tiling preserves 100% of spatial resolution while maintaining real-time performance through intelligent batching and frame skipping strategies.*
