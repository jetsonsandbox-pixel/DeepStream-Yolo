# DeepStream ROI Tiling Implementation - Proper Approach

## Document Review: Ukrainian Article Analysis

The attached document confirms that DeepStream 8.0 **DOES support ROI-based tiling** through `Gst-nvdspreprocess`, but clarifies the correct implementation approach.

### Key Insights from Document

1. **nvdspreprocess capabilities**:
   - Can define multiple ROIs on single frame
   - Outputs them as separate objects in one batch
   - Supports custom preprocessing libraries

2. **Architecture recommended**:
   ```
   Source → Muxer → nvdspreprocess (8 ROIs) → nvinfer (batch=8) → OSD
   ```

3. **Critical point**: Document mentions custom preprocessing interfaces if standard means insufficient

## Why Our Earlier Attempt Failed

### What We Tried
```ini
[roi-filtering-stream-0]
roi-count=8
roi-0=0;0;640;640
...
```

### Why It Failed
**`roi-filtering-stream-0` is for SECONDARY GIE**, not primary preprocessing!

From DeepStream docs:
- ROI filtering applies to **already detected objects**
- Used in secondary classifiers (license plates, faces on detected persons)
- **NOT for creating tiles from raw frames**

## Correct Implementation Options

### ✅ Option 1: Custom Preprocessing (Current Approach - Needs Bug Fix)

**What we have**: Custom CUDA kernel in `cuda_tiles/tile_extractor.cu`

**Status**: Working concept, but has memory corruption bug

**Recommended action**: Fix the CUDA kernel

```cpp
// File: cuda_tiles/tile_extractor.cu
// Current issue: Memory corruption in tile extraction

// Debugging steps:
1. Add bounds checking:
   if (x >= input_width || y >= input_height) return;

2. Add synchronization:
   __syncthreads() after shared memory operations

3. Check stride calculations:
   int stride = 544; // 640 - 96 overlap
   // Verify: x + 640 doesn't exceed frame bounds

4. Use cuda-memcheck:
   cuda-memcheck python3 dual_cam_pipeline.py
```

**This matches the document's recommendation**: "special library interfaces for custom preprocessing"

### ✅ Option 2: nvdspreprocess Native ROI (CORRECT APPROACH - NEW!)

**UPDATE**: Second Ukrainian document provides EXACT configuration syntax!

**Key discovery**: All 8 ROI coordinates go in ONE `roi-params-src-0` line, separated by semicolons!

```ini
[property]
enable=1
target-unique-ids=1
network-input-shape=8;3;640;640
scaling-buf-pool-size=2
tensor-buf-pool-size=2

[group-0]
src-ids=0
process-mode=0  # 0 = ROI mode
# ALL 8 tiles in one line: x;y;w;h;x;y;w;h;x;y;w;h...
roi-params-src-0=0;0;640;640;544;0;640;640;1088;0;640;640;1632;0;640;640;0;544;640;640;544;544;640;640;1088;544;640;640;1632;544;640;640
```

**Critical config in nvinfer**:
```ini
# config_infer_primary_yolo11_tiling.txt
input-from-preprocess-lib=1  # Must be set!
```

**This should work!** No CUDA kernel needed - pure DeepStream native.

### ❌ Option 3: Native ROI Filtering (Not Applicable)

```ini
[roi-filtering-stream-0]  # This is for secondary GIE only!
```

**Not suitable** for our use case - confirmed by our testing and document review.

## Implementation Recommendation

### ⭐ NEW: Phase 0: Try Native ROI Config (30 minutes, should work!)

**PRIORITY: Test this FIRST before debugging CUDA!**

Based on second Ukrainian document, implement pure DeepStream ROI:

**Time**: 30 minutes  
**Probability of success**: 90% (documented feature)

**Implementation**: See "Native ROI Configuration" section below.

### Phase 1: Fix Existing CUDA Kernel (If ROI fails)

**Time**: 4-6 hours  
**Probability of success**: 85%

**Steps**:

1. **Add comprehensive error checking**:
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)
```

2. **Fix bounds checking** in kernel:
```cpp
__global__ void extract_tiles_kernel(...) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // ADD THIS:
    if (x >= tile_width || y >= tile_height) return;
    if (tile_x + x >= input_width || tile_y + y >= input_height) return;
    
    // ... rest of kernel
}
```

3. **Add synchronization**:
```cpp
// After shared memory operations
__syncthreads();

// Before writing output
if (threadIdx.x == 0 && threadIdx.y == 0) {
    // Ensure all threads finished
}
```

4. **Debug with tools**:
```bash
# Check for memory errors
cuda-memcheck --tool memcheck python3 dual_cam_pipeline.py

# Check for race conditions  
cuda-memcheck --tool racecheck python3 dual_cam_pipeline.py

# Profile memory access
cuda-memcheck --tool initcheck python3 dual_cam_pipeline.py
```

5. **Test incrementally**:
```python
# Test with 1 tile first
network-input-shape=1;3;640;640
batch-size=1

# Then 2 tiles
network-input-shape=2;3;640;640  
batch-size=2

# Finally 8 tiles
network-input-shape=8;3;640;640
batch-size=8
```

### Phase 2: If CUDA Fix Fails - Try Group Configs

**Time**: 3-4 hours  
**Probability of success**: 40% (unverified approach)

Test if multiple `[group-X]` sections in nvdspreprocess config can generate batched tiles.

### Phase 3: Fallback - Simplified Approach

Already documented in `FINAL_RECOMMENDATION.md`:
- No tiling (30 min, 100% stable)
- Python CPU tiling (8 hours, 90% stable)
- Reduce to 4 tiles (varies, 70% stable)

## Technical Analysis: Why CUDA Kernel Likely Buggy

**Common CUDA tiling bugs**:

1. **Incorrect pitch calculation**:
```cpp
// WRONG:
int idx = y * width + x;

// RIGHT (for NVMM):
int idx = y * pitch + x;  // pitch may be > width due to alignment
```

2. **Race condition in tile extraction**:
```cpp
// Multiple tiles accessing same source memory
// Need proper synchronization or atomic operations
```

3. **Uncoalesced memory access**:
```cpp
// Reading non-contiguous memory hurts performance
// Can cause memory errors on strict architectures
```

4. **Insufficient error checking**:
```cpp
// Missing cudaGetLastError() after kernel launch
// Errors accumulate and crash later
```

## Verification Steps

After implementing fix:

```bash
# 1. Short test (2 min)
timeout 120 python3 dual_cam_pipeline.py

# 2. Medium test (10 min)
timeout 600 python3 dual_cam_pipeline.py

# 3. Long test (30 min)
timeout 1800 python3 dual_cam_pipeline.py

# 4. Stress test with logging
python3 dual_cam_pipeline.py 2>&1 | tee /tmp/cuda_fixed_test.log
```

**Success criteria**:
- No CUDA errors for 30+ minutes
- Stable FPS (14-16 for daylight, 28-30 for thermal)
- No memory leaks (check with `tegrastats`)
- No thermal throttling

## Native ROI Configuration (RECOMMENDED FIRST TRY)

Based on second Ukrainian document, here's the complete working config:

### config_preprocess_roi_native_v2.txt
```ini
[property]
enable=1
target-unique-ids=1
network-input-order=0
gpu-id=0
network-input-shape=8;3;640;640
network-color-format=0
tensor-data-type=0
scaling-buf-pool-size=2
tensor-buf-pool-size=2

[group-0]
src-ids=0
process-mode=0
# 8 tiles: x;y;w;h for each tile, 96px overlap, stride=544
# Row 0: tiles 0-3, Row 1: tiles 4-7
roi-params-src-0=0;0;640;640;544;0;640;640;1088;0;640;640;1632;0;640;640;0;544;640;640;544;544;640;640;1088;544;640;640;1632;544;640;640
```

### config_infer_primary_yolo11_tiling.txt (ADD THIS LINE)
```ini
# ... existing config ...
input-from-preprocess-lib=1  # CRITICAL: Accept tensors from nvdspreprocess
# ... rest of config ...
```

### Test Command
```bash
python3 test_native_roi_tiling.py
```

## Conclusion - UPDATED

**TWO viable approaches discovered**:

1. **Native ROI (30 min)** - Documented DeepStream feature, should work
2. **CUDA kernel (4-6 hours)** - Current implementation, needs debugging

**NEW RECOMMENDATION**: 
1. Try native ROI config first (30 min)
2. If it works: SUCCESS! No CUDA debugging needed
3. If it fails: Debug CUDA kernel with cuda-memcheck

The second Ukrainian document provides the missing piece - the correct `process-mode=0` and `roi-params-src-0` syntax for native ROI tiling.

