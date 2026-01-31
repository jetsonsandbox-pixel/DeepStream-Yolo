# CUDA Kernel Debugging Plan

## Reality Check: Native ROI Isn't the Solution

After reviewing official DeepStream documentation and sample configs, **native ROI support is NOT designed for tile-based batching**.

**What ROI does**:
- Extracts specific regions from frames
- Useful for multi-stream with different ROIs per stream
- Requires custom transformation function
- **Does NOT automatically batch multiple ROIs as separate inference inputs**

**Our CUDA kernel approach IS the correct solution** - we just need to fix the memory bug.

## CUDA Kernel Location

```bash
cuda_tiles/tile_extractor.cu
nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
```

## Debugging Steps

### Step 1: Run with cuda-memcheck (30 min)

```bash
cuda-memcheck --tool memcheck python3 dual_cam_pipeline.py 2>&1 | tee /tmp/cuda_memcheck.log
```

**What to look for**:
- Invalid global read/write
- Out of bounds access
- Uninitialized memory
- Race conditions

### Step 2: Check Kernel Launch Parameters

```cpp
// In tile_extractor.cu
__global__ void extract_tiles_kernel(...)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // CHECK 1: Bounds checking
    if (x >= tile_width || y >= tile_height) return;
    
    // CHECK 2: Source bounds
    int src_x = tile_offset_x + x;
    int src_y = tile_offset_y + y;
    if (src_x >= input_width || src_y >= input_height) return;
    
    // CHECK 3: Proper pitch calculation
    int src_idx = src_y * input_pitch + src_x;  // NOT input_width!
    int dst_idx = y * tile_width + x;
    
    // Copy pixel
    output[dst_idx] = input[src_idx];
}
```

### Step 3: Add Error Checking

```cpp
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return err; \
    } \
} while(0)

// After kernel launch:
extract_tiles_kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

### Step 4: Common CUDA Tiling Bugs

**Bug 1: Incorrect pitch calculation**
```cpp
// WRONG:
int idx = y * width + x;

// RIGHT (NVMM surfaces have alignment):
int idx = y * pitch + x;
```

**Bug 2: Tile overlap causing race conditions**
```cpp
// With 96px overlap, multiple tiles access same source pixels
// Ensure READ-ONLY access, no writes to source
```

**Bug 3: Insufficient bounds checking**
```cpp
// Check both source AND destination bounds
// Last tiles (3, 7) extend beyond 1920×1080
// Need padding or clipping
```

**Bug 4: Block/Grid size mismatch**
```cpp
dim3 block(16, 16);  // 256 threads
dim3 grid((tile_width + 15) / 16, (tile_height + 15) / 16);

// Verify grid covers entire tile
// Verify no out-of-bounds threads
```

##  Step 5: Incremental Testing

```bash
# Test 1 tile
sed -i 's/network-input-shape=8/network-input-shape=1/' config_preprocess_tiling.txt
sed -i 's/batch-size=8/batch-size=1/' config_infer_primary_yolo11_tiling.txt
python3 test_daylight_only.py

# Test 2 tiles
sed -i 's/network-input-shape=1/network-input-shape=2/' config_preprocess_tiling.txt
sed -i 's/batch-size=1/batch-size=2/' config_infer_primary_yolo11_tiling.txt
python3 test_daylight_only.py

# Test 4 tiles
# ... etc
```

### Step 6: Check Kernel Source

```bash
# Find the actual implementation
find nvdsinfer_custom_impl_Yolo -name "*.cu" -o -name "*.cpp" | xargs grep -l "tile"
cat cuda_tiles/tile_extractor.cu
```

## Expected Fixes

Based on typical CUDA bugs:

1. **Add synchronization**:
```cpp
__syncthreads();  // After shared memory operations
```

2. **Fix bounds**:
```cpp
if (src_x + tile_width > input_width) {
    // Pad or clip
}
```

3. **Use proper pitch**:
```cpp
NvBufSurfaceParams *surf = &surface->surfaceList[0];
int pitch = surf->pitch;  // Use this, not width!
```

4. **Add error checking everywhere**:
```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    g_print("Kernel failed: %s\\n", cudaGetErrorString(err));
}
```

## Timeline

- **Step 1-2**: 1 hour (cuda-memcheck + analysis)
- **Step 3**: 30 min (add error checking)
- **Step 4-5**: 2 hours (fix bugs + test incrementally)
- **Step 6**: 30 min (verify fix with long run)

**Total**: 4 hours

## Success Criteria

- No CUDA errors in cuda-memcheck
- Stable for 30+ minutes
- FPS ~14-16 (daylight) and ~28-30 (thermal)
- No memory leaks
- No thermal throttling

## Next Action

```bash
cd /home/jet-nx8/DeepStream-Yolo
cuda-memcheck --tool memcheck python3 dual_cam_pipeline.py 2>&1 | tee /tmp/cuda_debug.log
```

Then analyze the output for exact line numbers and error types.

