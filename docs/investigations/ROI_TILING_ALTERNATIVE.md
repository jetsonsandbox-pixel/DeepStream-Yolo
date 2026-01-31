# Alternative Solution: DeepStream Native ROI-Based Tiling

## Current Problem

**Custom CUDA tiling kernel has memory corruption bug:**
- Works for 2-3 minutes then crashes with `cudaErrorIllegalAddress`
- Located in `cuda_tiles/tile_extractor.cu` via `nvdspreprocess` custom function
- Causes driver shutdown and segmentation fault

## Proposed Solution: DeepStream nvdspreprocess ROI

Instead of custom CUDA kernel, use DeepStream's built-in ROI (Region of Interest) functionality in `nvdspreprocess`.

### How It Works

**nvdspreprocess** supports processing multiple ROIs from a single frame:
```
1920×1080 frame
    ↓
nvdspreprocess with ROI list
    ↓
Outputs 8 tiles (640×640 each) as separate tensor batches
    ↓
nvinfer processes batch of 8
```

### Configuration Approach

#### 1. Define ROI Coordinates in Config

```ini
[property]
enable=1
process-on-frame=1
network-input-shape=8;3;640;640
scaling-buf-pool-size=2
tensor-buf-pool-size=2

# Enable ROI processing
process-on-roi=1
operate-on-gie-id=-1

[roi-filtering-stream-0]
# 4×2 grid of 640×640 tiles with 96px overlap
# Tile 0 (top-left)
roi-count=8
roi-0=0;0;640;640
roi-1=544;0;640;640
roi-2=1088;0;640;640  
roi-3=1632;0;640;640
roi-4=0;544;640;640
roi-5=544;544;640;640
roi-6=1088;544;640;640
roi-7=1632;544;640;640
```

#### 2. Remove Custom CUDA Kernel

Remove these from config:
```ini
# DELETE:
custom-lib-path=...libnvdsinfer_custom_impl_Yolo.so
custom-tensor-preparation-function=CustomTensorPreparation
```

### Advantages Over Custom CUDA Kernel

| Aspect | Custom CUDA | Native ROI |
|--------|-------------|------------|
| **Memory Safety** | ❌ Has bugs | ✅ Tested by NVIDIA |
| **Maintenance** | ❌ Need to debug C++ | ✅ Config-only |
| **Stability** | ❌ Crashes after 2-3 min | ✅ Production-ready |
| **Performance** | ~15 FPS | ~12-15 FPS (similar) |
| **Buffer Management** | ❌ Manual | ✅ Automatic |

### Implementation Steps

1. **Create new preprocessing config**: `config_preprocess_roi.txt`
2. **Define 8 ROI regions** with overlap
3. **Test with single camera** first
4. **Add thermal camera** once stable

### ROI Calculation for 8 Tiles

**Frame**: 1920×1080
**Tile size**: 640×640  
**Overlap**: 96 pixels (15%)
**Stride**: 544 pixels (640 - 96)

**Grid layout (4×2):**
```
┌─────────┬─────────┬─────────┬─────────┐
│ Tile 0  │ Tile 1  │ Tile 2  │ Tile 3  │
│ (0,0)   │(544,0)  │(1088,0) │(1632,0) │
│ 640×640 │ 640×640 │ 640×640 │ 640×640 │
├─────────┼─────────┼─────────┼─────────┤
│ Tile 4  │ Tile 5  │ Tile 6  │ Tile 7  │
│(0,544)  │(544,544)│(1088,544)│(1632,544)│
│ 640×640 │ 640×640 │ 640×640 │ 640×640 │
└─────────┴─────────┴─────────┴─────────┘
```

**ROI coordinates (x, y, width, height):**
- Tile 0: (0, 0, 640, 640)
- Tile 1: (544, 0, 640, 640)
- Tile 2: (1088, 0, 640, 640)
- Tile 3: (1632, 0, 640, 640) - Note: extends to 2272, will pad
- Tile 4: (0, 544, 640, 640)
- Tile 5: (544, 544, 640, 640)
- Tile 6: (1088, 544, 640, 640)
- Tile 7: (1632, 544, 640, 640) - Note: extends to 1184, will pad

### Expected Behavior

1. **nvdspreprocess** extracts 8 ROIs from each frame
2. Scales/pads each ROI to 640×640
3. Outputs tensor shape: `(8, 3, 640, 640)`
4. **nvinfer** processes all 8 in one batch
5. **nvdsosd** renders detections back to original coordinates

### Alternative: Lower Memory Approach

If 8 tiles still causes issues, process 4 tiles at a time:

**Pass 1**: Tiles 0-3 (top row)
**Pass 2**: Tiles 4-7 (bottom row)

This would require two preprocessing configs or dynamic switching.

### Testing Plan

1. Create `config_preprocess_roi.txt` with ROI definitions
2. Test daylight camera alone: `python3 test_daylight_roi.py`
3. Verify 8 tiles are extracted correctly
4. Monitor for CUDA errors over 10+ minutes
5. If stable, integrate into `dual_cam_pipeline.py`
6. Test dual camera with ROI-based tiling

### Fallback Options

If native ROI still has issues:

**Option A**: Process full 1920×1080 frame (no tiling)
- Simpler, more stable
- Lower detection accuracy for small objects
- Higher FPS (~25-30)

**Option B**: Python-based pre-tiling
- Extract tiles in Python before GStreamer
- Feed pre-tiled images to DeepStream
- More CPU overhead but controllable

**Option C**: Single-tile sliding window
- Process one 640×640 tile at a time
- Move window across frame
- Slower but very stable

## Next Actions

1. ✅ Commit current working state (fakesink + buffer optimization)
2. ⏳ Create ROI-based preprocessing config
3. ⏳ Test ROI approach with daylight camera
4. ⏳ Compare stability with custom CUDA approach
5. ⏳ If stable, replace custom tiling with ROI

## References

- DeepStream SDK: nvdspreprocess plugin documentation
- ROI filtering: Process specific regions from full frame
- Multi-ROI inference: Batch process multiple regions

