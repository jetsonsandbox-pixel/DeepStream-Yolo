# Dual Camera Tiling Investigation Summary

## Current Status (Committed)

**Branch**: `feature/dual-camera`  
**Commit**: `0da168f` - "Fix: Dual camera CUDA memory issues with fakesink and buffer optimization"

**Working Configuration:**
- Daylight CSI (1920×1080) with 8-tile tiling → model_b8_gpu0_fp16.engine
- Thermal USB (640×512) no tiling → model_thermal_b1_gpu0_fp16.engine
- Sinks: fakesink (no display - saves GPU memory)
- Buffer pools: 2 (most stable)
- Queue management: leaky queues with max 2 buffers
- **Stability**: ~3 minutes before CUDA crash

**Known Issue:**
- Custom CUDA tiling kernel (`cuda_tiles/tile_extractor.cu`) has memory corruption bug
- Crashes with `cudaErrorIllegalAddress (700)` after 2-3 minutes
- Error: "Tile extraction kernel launch failed: driver shutting down"

## Investigation Results

### Approach 1: DeepStream Native ROI ❌ Failed

**Attempted**: Use nvdspreprocess with ROI filtering
- **File**: `config_preprocess_roi_native.txt`
- **Result**: Pipeline fails to start
- **Error**: "Some preprocess config properties not set"
- **Conclusion**: nvdspreprocess ROI feature doesn't support multi-tile batching as we need it

**Why it doesn't work:**
- ROI filtering in nvdspreprocess is designed for filtering objects, not creating multiple output tiles
- Cannot output batch of 8 tiles from single frame via config alone
- Would need custom preprocessing plugin anyway

### Approach 2: Python-Based Tiling ⏳ Recommended

**Concept**: Extract tiles in Python before GStreamer pipeline

**Architecture:**
```
Camera (1920×1080)
    ↓
Python: extract_tiles() → 8 tiles (640×640)
    ↓
8× appsrc elements (one per tile)
    ↓
nvstreammux (batch=8)
    ↓
nvinfer (batch=8 model)
    ↓
nvdsosd → fakesink
```

**Advantages:**
- ✅ **Memory safe**: No CUDA kernel bugs
- ✅ **Debuggable**: Pure Python, easy to trace
- ✅ **Stable**: No driver crashes
- ✅ **Maintainable**: No C++/CUDA expertise needed
- ✅ **Flexible**: Easy to adjust overlap, padding, grid

**Performance Trade-offs:**
- CPU tile extraction: ~5-10ms per frame
- memcpy CPU→GPU: ~2-3ms  
- **Expected FPS**: 20-25 (vs 15 with buggy CUDA)
- **Throughput**: Actually better than crashing!

**Implementation Complexity**: Medium
- Need to manage 8 appsrc elements
- Need to map detection coordinates back to original frame
- More pipeline plumbing code

### Approach 3: GStreamer Native videocrop ⏳ Alternative

**Concept**: Use GStreamer's videocrop + videoscale plugins

**Architecture:**
```
nvarguscamerasrc
  ├─> videocrop (0,0,640,640) → nvstreammux.sink_0
  ├─> videocrop (544,0,640,640) → nvstreammux.sink_1
  ├─> videocrop (1088,0,640,640) → nvstreammux.sink_2
  ├─> videocrop (1632,0,640,640) → nvstreammux.sink_3
  ├─> videocrop (0,440,640,640) → nvstreammux.sink_4
  ├─> videocrop (544,440,640,640) → nvstreammux.sink_5
  ├─> videocrop (1088,440,640,640) → nvstreammux.sink_6
  └─> videocrop (1632,440,640,640) → nvstreammux.sink_7
```

**Advantages:**
- ✅ All native GStreamer
- ✅ GPU-accelerated (VIC engine)
- ✅ No custom code
- ✅ Very stable

**Disadvantages:**
- ❌ Complex pipeline (8 branches)
- ❌ Higher memory usage
- ❌ Uses tee element (more overhead)

**Implementation Complexity**: Medium-High
- 8 parallel branches
- Need to synchronize streams
- More memory management

### Approach 4: No Tiling (Simplest) 💡 Fallback

**Concept**: Process full 1920×1080 frame without tiling

**Changes:**
- Use different model trained on 1920×1080 or resize to single 640×640
- No tile extraction needed
- Much simpler pipeline

**Advantages:**
- ✅ Extremely stable
- ✅ Simplest implementation
- ✅ Lower memory usage
- ✅ Higher FPS (~25-30)

**Disadvantages:**
- ❌ Lower detection accuracy for small objects
- ❌ Need to retrain model or accept resized input

## Recommendation

### Phase 1: Quick Win - GStreamer videocrop Approach

**Reason**: Fastest to implement with good stability

**Implementation** (est. 2-3 hours):
1. Create pipeline with tee + 8× videocrop branches
2. Connect to nvstreammux (batch=8)
3. Test stability for 10+ minutes
4. Integrate thermal camera

**Risk**: Medium memory usage, but likely stable

### Phase 2: If videocrop fails - Python Tiling

**Reason**: Most control and debuggability

**Implementation** (est. 4-6 hours):
1. Implement Python tile extraction function
2. Create 8× appsrc pipeline
3. Push tiles from Python to GStreamer
4. Map detection coordinates back
5. Test stability

**Risk**: CPU overhead, but calculable and acceptable

### Phase 3: Production - No Tiling (if detection accuracy acceptable)

**Reason**: Simplest and most stable long-term

**Implementation** (est. 1 hour):
1. Remove tiling configuration
2. Use model_thermal_b1_gpu0_fp16.engine for both cameras (or train new full-res model)
3. Test detection accuracy
4. If acceptable, deploy

**Risk**: Lower small object detection, but may be acceptable for use case

## Next Actions

1. **Immediate**: Implement GStreamer videocrop approach (test_videocrop_tiling.py)
2. **Fallback**: Python tiling if videocrop unstable
3. **Long-term**: Evaluate if tiling is truly necessary for detection requirements

## Files Created

- [docs/ROI_TILING_ALTERNATIVE.md](docs/ROI_TILING_ALTERNATIVE.md) - ROI investigation
- [docs/PYTHON_TILING_APPROACH.md](docs/PYTHON_TILING_APPROACH.md) - Python approach details
- [config_preprocess_roi_native.txt](config_preprocess_roi_native.txt) - Failed ROI config (kept for reference)
- [test_roi_tiling.py](test_roi_tiling.py) - Failed ROI test
- [test_python_tiling.py](test_python_tiling.py) - Python approach starter

## Decision Matrix

| Approach | Stability | Performance | Complexity | Time to Implement |
|----------|-----------|-------------|------------|-------------------|
| **Custom CUDA** | ❌ Crashes | 15 FPS | High | N/A (buggy) |
| **DeepStream ROI** | ❌ Doesn't work | N/A | N/A | N/A |
| **videocrop (GStreamer)** | ✅ Likely | 15-20 FPS | Medium | 2-3 hours |
| **Python Tiling** | ✅ Very likely | 20-25 FPS | Medium | 4-6 hours |
| **No Tiling** | ✅ Guaranteed | 25-30 FPS | Low | 1 hour |

**Recommendation**: Start with **videocrop**, validate stability. If issues, move to **Python tiling**. Consider **no tiling** if detection accuracy acceptable.

