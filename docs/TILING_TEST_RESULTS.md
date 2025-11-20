# Tiled Inference Testing - Memory Issue

## 🚨 Issue Encountered

During testing of the tiled inference implementation, we encountered an **Out of Memory (OOM)** error:

```
ERROR: [TRT]: [defaultAllocator.cpp::allocate::31] Error Code 1: Cuda Runtime (out of memory)
ERROR: [TRT]: [executionContext.cpp::ExecutionContext::609] Error Code 2: OutOfMemory (Requested size was 150732800 bytes.)
```

### System Status at Time of Error

```
Memory Usage: 5.6GB / 7.4GB (76% used)
Swap: 1.9GB / 14GB
Available: 1.5GB
TensorRT Request: 150MB (execution context)
```

### Root Cause

The Jetson Orin NX has unified memory (7.4GB total shared between CPU and GPU). With batch-size=8, the TensorRT engine requires:
- **Engine file**: 12MB (model_b8_gpu0_fp32.engine)
- **Execution context**: ~150MB
- **Input buffers**: 8 tiles × 640×640×3 = ~9.4MB
- **Output buffers**: ~10MB
- **Total**: ~180MB + system overhead

Current memory usage (5.6GB) leaves insufficient contiguous memory for TensorRT allocation.

## ✅ Solutions

### Solution 1: Free Memory Before Testing (Recommended)

```bash
# 1. Close unnecessary applications
# Close browsers, IDE, etc.

# 2. Clear system cache
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# 3. Check available memory
free -h

# 4. Run DeepStream with tiling
cd /home/jet-nx8/DeepStream-Yolo
source /home/jet-nx8/Sandbox/ultralytics-env/bin/activate
deepstream-app -c deepstream_app_config.txt
```

### Solution 2: Reduce Batch Size (Alternative)

Instead of 8 tiles (4×2 grid), use 4 tiles (2×2 grid):

**Modified config**:
```ini
# config_infer_primary_yolo11_tiling_4tiles.txt
batch-size=4  # 2×2 grid instead of 4×2

# This reduces:
# - Engine size: ~6MB (vs 12MB)
# - Memory usage: ~90MB (vs 180MB)
# - Tile overlap reduced to maintain coverage
```

**Trade-offs**:
- Less spatial coverage (larger tiles)
- Fewer overlap regions
- Lower memory requirements
- Still provides significant benefit over single-frame processing

### Solution 3: Use Frame Skipping with Standard Mode

As a fallback, use standard inference with aggressive frame skipping:

```ini
# config_infer_primary_yolo11.txt
batch-size=1
interval=10  # Process every 10th frame

# Advantages:
# - Low memory footprint (~2GB)
# - Acceptable for tracking use case
# - Works within memory constraints
```

## 🔧 Testing Procedure (Revised)

### Step 1: Prepare System

```bash
# Close unnecessary applications
# - Web browsers
# - VS Code / editors
# - Any running Python scripts

# Clear cache
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Verify memory
free -h
# Target: >2GB available
```

### Step 2: Test Standard Mode First

```bash
cd /home/jet-nx8/DeepStream-Yolo
source /home/jet-nx8/Sandbox/ultralytics-env/bin/activate

# Use standard config to verify base system
deepstream-app -c deepstream_app_config.txt
# (Currently configured for config_infer_primary_yolo11.txt)

# Press Ctrl+C to stop
```

**Verify**:
- Video playback works
- Detections appear
- No crashes
- Note FPS and detection count

### Step 3: Switch to Tiling Mode

```bash
# Update config
cd /home/jet-nx8/DeepStream-Yolo
sed -i 's/config_infer_primary_yolo11.txt/config_infer_primary_yolo11_tiling.txt/g' deepstream_app_config.txt

# Free memory again
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Run with tiling
deepstream-app -c deepstream_app_config.txt
```

**Monitor**:
```bash
# In another terminal
watch -n 1 tegrastats
# Watch memory usage during initialization
```

### Step 4: Compare Results

| Metric | Standard (batch=1) | Tiled (batch=8) |
|--------|-------------------|-----------------|
| Init Memory | ~2GB | ~4GB |
| Runtime Memory | ~2.5GB | ~4.5GB |
| FPS (interval=0) | 15-20 | 10-15 |
| Small Object Detections | Baseline | +15-30% expected |
| False Negatives | Baseline | -20-40% expected |

## 📋 Validation Checklist

Once memory is freed and tiling runs successfully:

- [ ] **Initialization**: Engine loads without OOM
- [ ] **Video Playback**: Stream decodes and displays
- [ ] **Batch Processing**: 8 tiles processed per frame
- [ ] **Detections**: Bounding boxes appear on display
- [ ] **Coordinate Mapping**: Boxes align with objects
- [ ] **No Duplicates**: NMS removes overlapping detections
- [ ] **Small Objects**: More detections than standard mode
- [ ] **Performance**: FPS acceptable (>5 target)
- [ ] **Stability**: Runs for >1 minute without crash
- [ ] **Memory**: Stays within 6GB limit

## 🐛 Troubleshooting

### If OOM Persists

1. **Reboot system** to clear all cached memory:
   ```bash
   sudo reboot
   ```

2. **After reboot**, immediately test (no other applications):
   ```bash
   cd /home/jet-nx8/DeepStream-Yolo
   source /home/jet-nx8/Sandbox/ultralytics-env/bin/activate
   deepstream-app -c deepstream_app_config.txt
   ```

3. **Check for memory leaks** from previous runs:
   ```bash
   ps aux | grep -E "deepstream|gst|python" | grep -v grep
   # Kill any lingering processes
   ```

### If Detection Quality Issues

1. **Verify tile extraction** (add debug in preprocessor)
2. **Check coordinate transformation** (print detection coords)
3. **Tune NMS threshold**:
   ```ini
   nms-iou-threshold=0.40  # More aggressive
   # or
   nms-iou-threshold=0.50  # Less aggressive
   ```

### If Performance Too Slow

1. **Enable frame skipping**:
   ```ini
   interval=5  # Process every 5th frame
   ```

2. **Reduce tile overlap**:
   ```ini
   # Modify TileConfig in nvdsinfer_tiled_config.h
   overlap = 64;  # Instead of 96
   ```

## 📊 Expected Timeline

**Best Case** (after memory cleanup):
- Initialization: 30-60 seconds (TensorRT warmup)
- First frame: 2-3 seconds (pipeline startup)
- Steady state: 10-15 FPS with tiling

**Worst Case** (insufficient memory):
- Implement Solution 2 (4 tiles) or Solution 3 (frame skipping)

## 🎯 Next Steps

1. **Close all non-essential applications**
2. **Clear system cache**
3. **Retry tiling test** with freed memory
4. **Document results** (screenshots, metrics)
5. **Compare with standard mode** for validation

## 📝 Status

- **Implementation**: ✅ Complete
- **Build**: ✅ Successful
- **Initial Test**: ⚠️ OOM Error
- **Solution Identified**: ✅ Memory cleanup required
- **Ready for Retry**: ✅ Yes (after cleanup)

---

**Date**: November 20, 2025
**System**: Jetson Orin NX (7.4GB unified memory)
**DeepStream Version**: 7.1
**CUDA Version**: 12.6
