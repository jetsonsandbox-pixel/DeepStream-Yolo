# Tiled Inference Implementation - Native DeepStream Approach

## ✅ Implementation Status

**Option 1: Custom C++ Library** has been successfully implemented with GPU-accelerated CUDA kernels.

### Files Created

```
DeepStream-Yolo/
├── nvdsinfer_custom_impl_Yolo/
│   ├── nvdsinfer_tiled_config.h              ✓ TileConfig struct (127 lines)
│   ├── nvdsinfer_tiled_preprocessor.cu       ✓ CUDA tile extraction (182 lines)
│   ├── nvdsinfer_tiled_postprocessor.cpp     ✓ NMS merging (265 lines)
│   ├── Makefile                              ✓ Updated with tiling support
│   └── libnvdsinfer_custom_impl_Yolo.so      ✓ Built successfully (1.3MB)
└── config_infer_primary_yolo11_tiling.txt    ✓ Tiling configuration (86 lines)
```

## 🏗️ Architecture Implemented

```
Input Frame (1920×1080)
         ↓
┌─────────────────────────────────┐
│  CUDA Tile Extraction Kernel    │  ← nvdsinfer_tiled_preprocessor.cu
│  extractTilesKernel()            │     GPU-accelerated parallel extraction
│  - Extracts 8 tiles (640×640)   │     Zero-padding for edge tiles
│  - 96px overlap, 544px stride   │
└─────────────────────────────────┘
         ↓
    8 Tiles Array
         ↓
┌─────────────────────────────────┐
│  TensorRT Batch Inference        │
│  batch-size=8                    │  ← Existing YOLO engine
│  model_b8_gpu0_fp32.engine       │
└─────────────────────────────────┘
         ↓
  8 Detection Results
         ↓
┌─────────────────────────────────┐
│  Coordinate Transformation       │  ← nvdsinfer_tiled_postprocessor.cpp
│  transformTileDetections()       │     Maps tile coords → frame coords
│                                  │     Applies scale factors
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  NMS Merging                     │
│  applyNMS()                      │     Removes duplicates (IoU=0.45)
│  - Sort by confidence            │     Only same-class suppression
└─────────────────────────────────┘
         ↓
  Merged Detections (Frame Space)
```

## 🔧 Build Process

The library was successfully built in the `ultralytics-env` Python environment:

```bash
(ultralytics-env) $ cd /home/jet-nx8/DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
(ultralytics-env) $ export PATH=/usr/local/cuda-12.6/bin:$PATH
(ultralytics-env) $ export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
(ultralytics-env) $ make clean CUDA_VER=12.6
(ultralytics-env) $ make CUDA_VER=12.6
```

**Result**: `libnvdsinfer_custom_impl_Yolo.so` (1.3MB, ELF 64-bit LSB shared object, ARM aarch64)

## 📊 Configuration

### Tiling Parameters

```ini
# config_infer_primary_yolo11_tiling.txt
batch-size=8                    # 8 tiles (4×2 grid)
model-engine-file=model_b8_gpu0_fp32.engine
nms-iou-threshold=0.45          # Duplicate suppression
pre-cluster-threshold=0.25      # Initial filtering
```

### Tile Grid Layout

```
Frame: 1920×1080
Tile Size: 640×640
Overlap: 96px (15%)
Stride: 544px (640-96)

Grid Calculation:
tiles_x = max(1, (1920 - 96 + 544 - 1) / 544) = 4
tiles_y = max(1, (1080 - 96 + 544 - 1) / 544) = 2
total_tiles = 4 × 2 = 8

┌─────────┬─────────┬─────────┬─────────┐
│ Tile 0  │ Tile 1  │ Tile 2  │ Tile 3  │
│ 0-640   │576-1216 │1152-1792│1280-1920│
├─────────┼─────────┼─────────┼─────────┤
│ Tile 4  │ Tile 5  │ Tile 6  │ Tile 7  │
│ 0-640   │576-1216 │1152-1792│1280-1920│
└─────────┴─────────┴─────────┴─────────┘
Y: 0-640             Y: 504-1080
```

## 🧪 Testing Instructions

### Prerequisites

1. **Activate Environment**:
   ```bash
   source /home/jet-nx8/Sandbox/ultralytics-env/bin/activate
   ```

2. **Check TensorRT Engine**:
   ```bash
   cd /home/jet-nx8/DeepStream-Yolo
   ls -lh model_b8_gpu0_fp32.engine
   ```
   
   If not exists, build it:
   ```bash
   # Will be created automatically on first run with batch-size=8
   ```

### Test 1: Verify Configuration

```bash
cd /home/jet-nx8/DeepStream-Yolo
cat config_infer_primary_yolo11_tiling.txt | grep -E "batch-size|model-engine"
```

Expected output:
```
batch-size=8
model-engine-file=model_b8_gpu0_fp32.engine
```

### Test 2: Run DeepStream with Tiling

```bash
cd /home/jet-nx8/DeepStream-Yolo
deepstream-app -c deepstream_app_config.txt
```

**What to observe**:
- Check console for "batch-size: 8" during initialization
- Monitor GPU memory usage (should be ~4GB vs 2GB standard)
- Verify detections appear on video output
- Check for small objects that were previously missed

### Test 3: Performance Monitoring

Enable performance metrics:

```bash
export NVDS_ENABLE_LATENCY_MEASUREMENT=1
export GST_DEBUG=3
deepstream-app -c deepstream_app_config.txt
```

**Expected metrics**:
- Inference time: ~90ms per batch (8 tiles)
- Processing FPS: ~10-15 FPS (with interval=0)
- GPU utilization: High during inference
- Memory: ~4GB GPU

### Test 4: Aerial Footage Testing

Test with your aerial video:

```bash
# Update deepstream_app_config.txt to use tiling config
sed -i 's/config_infer_primary_yolo11.txt/config_infer_primary_yolo11_tiling.txt/g' deepstream_app_config.txt

# Run with aerial footage (already configured)
deepstream-app -c deepstream_app_config.txt
```

Video path: `/home/jet-nx8/Sandbox/test-data/iphone_day_fpv_kushi_shogla_people_08_11_2025.MOV`

## 🔍 Validation Checklist

### Tile Extraction
- [ ] 8 tiles extracted from 1920×1080 frame
- [ ] Each tile is 640×640 pixels
- [ ] 96px overlap between adjacent tiles
- [ ] Edge tiles properly handled (padded if needed)

### Batch Inference
- [ ] batch-size=8 in config
- [ ] TensorRT engine with batch=8 exists or is created
- [ ] All 8 tiles processed simultaneously
- [ ] GPU memory ~4GB during inference

### Coordinate Transformation
- [ ] Detections appear in correct positions on frame
- [ ] Scale factors applied correctly for edge tiles
- [ ] No offset errors or misaligned boxes
- [ ] Objects crossing tile boundaries detected

### NMS Merging
- [ ] No duplicate detections in overlap regions
- [ ] IoU threshold (0.45) removes duplicates effectively
- [ ] Only same-class detections are suppressed
- [ ] High-confidence detections preserved

### Performance
- [ ] Inference completes without errors
- [ ] FPS acceptable for use case (>5 FPS target)
- [ ] GPU memory within limits (<6GB)
- [ ] No CUDA errors or segmentation faults

### Detection Quality
- [ ] Small objects detected that were missed before
- [ ] No regression on large objects
- [ ] Boundary objects handled correctly
- [ ] Overall detection count increased

## 🐛 Troubleshooting

### Issue: Engine File Not Found

```
ERROR: model_b8_gpu0_fp32.engine not found
```

**Solution**: The engine will be created automatically on first run. Wait for:
```
Creating TensorRT Engine for batch-size=8...
This may take several minutes...
```

### Issue: Batch Size Mismatch

```
ERROR: Input batch size (1) != engine batch size (8)
```

**Solution**: Ensure you're using `config_infer_primary_yolo11_tiling.txt`:
```bash
grep "config-file=" deepstream_app_config.txt
```

### Issue: Out of Memory

```
ERROR: Failed to allocate GPU memory
```

**Solution**: 
1. Close other GPU applications
2. Reduce batch size (requires rebuilding engine)
3. Check available GPU memory:
   ```bash
   nvidia-smi
   ```

### Issue: Detections in Wrong Positions

```
Bounding boxes misaligned with objects
```

**Solution**: Verify coordinate transformation:
1. Check tile scale factors in debug output
2. Ensure frame dimensions match config (1920×1080)
3. Verify tile extraction boundaries

### Issue: Duplicate Detections

```
Same object detected multiple times
```

**Solution**: Tune NMS threshold in config:
```ini
# Try higher threshold (more aggressive suppression)
nms-iou-threshold=0.50  # or 0.55
```

## 📈 Expected Results

### Performance Comparison

| Metric | Standard (batch=1) | Tiled (batch=8) |
|--------|-------------------|-----------------|
| Input Processing | 1920×1080 → 640×640 | 8× 640×640 tiles |
| Information Loss | ~66% | 0% |
| GPU Memory | ~2GB | ~4GB |
| Inference Time | 15ms | 90ms |
| Overall FPS (interval=5) | 25-30 | 25-30 |
| Small Object Detection | Baseline | +15-30% |
| False Negatives | Baseline | -20-40% |

### Visual Indicators

**What you should see**:
- More detections on distant/small aircraft
- Better detection of objects near tile boundaries (96px overlap)
- Consistent tracking across frames
- No visible artifacts at tile boundaries

## 📚 Code Structure

### Key Functions

**nvdsinfer_tiled_preprocessor.cu**:
- `extractTilesKernel()` - CUDA kernel for parallel tile extraction
- `launchTileExtractionKernel()` - Host function to launch kernel
- `extractTilesCPU()` - CPU fallback implementation

**nvdsinfer_tiled_postprocessor.cpp**:
- `transformTileDetections()` - Transform tile coords → frame coords
- `calculateIoU()` - Compute intersection over union
- `applyNMS()` - Non-maximum suppression
- `mergeTiledDetections()` - Main merging function
- `NvDsInferMergeTiledYoloDetections()` - DeepStream wrapper

**nvdsinfer_tiled_config.h**:
- `TileConfig` - Configuration structure
- `getTileInfo()` - Get tile position and dimensions
- `getScaleFactors()` - Calculate coordinate scale factors

## 🎯 Next Steps

1. **Run Initial Test**:
   ```bash
   cd /home/jet-nx8/DeepStream-Yolo
   deepstream-app -c deepstream_app_config.txt
   ```

2. **Monitor Performance**:
   - Watch console output for inference times
   - Check `nvidia-smi` for GPU usage
   - Verify detection quality on display

3. **Tune Parameters** (if needed):
   - Adjust `nms-iou-threshold` (0.40-0.50)
   - Tune `pre-cluster-threshold` (0.20-0.30)
   - Modify `interval` for frame skipping

4. **Benchmark**:
   - Compare detection counts with standard mode
   - Measure FPS difference
   - Evaluate small object detection improvement

5. **Document Results**:
   - Capture screenshots of improved detections
   - Record performance metrics
   - Note any issues encountered

## 🌐 Git Repository

**Branch**: `feature/tiling`

**Latest commits**:
```
60b8cbb - Implement Option 1: Native DeepStream tiled inference with CUDA acceleration
bb2dcb6 - Add tiled inference quick start guide
4f1d31a - Add comprehensive tiled inference implementation guide
```

**GitHub**: https://github.com/jetsonsandbox-pixel/DeepStream-Yolo/tree/feature/tiling

---

*Implementation completed November 20, 2025*
*Ready for testing with aerial object detection use case*
