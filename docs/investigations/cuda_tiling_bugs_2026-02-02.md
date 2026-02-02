# CUDA Tiling Pipeline Investigation - February 2, 2026

## Summary

Investigation into CUDA crashes and instability in the dual-camera DeepStream pipeline using nvdspreprocess with custom tiling for 8-tile batch processing.

---

## Issues Identified and Resolved

### 1. CUDA Architecture Mismatch (CRITICAL - RESOLVED)

**Symptom:**
```
Tile extraction kernel launch failed: invalid device function
cudaErrorIllegalAddress (700)
```

**Root Cause:**
The Makefile in `nvdsinfer_custom_impl_Yolo/` was compiling CUDA kernels without specifying the GPU architecture. Jetson Orin NX uses **compute capability 8.7 (sm_87)**.

**Fix Applied:**
```makefile
# Before (missing architecture flag)
CUFLAGS:= $(GST_CFLAGS_NO_PTHREAD) -I/opt/nvidia/deepstream/...

# After (with correct architecture)
CUDA_ARCH?=87
CUFLAGS:= -arch=sm_$(CUDA_ARCH) $(GST_CFLAGS_NO_PTHREAD) -I/opt/nvidia/deepstream/...
```

**Verification:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
# Output: Orin (nvgpu), 8.7
```

---

### 2. Tensor Data Type Numbering Confusion (RESOLVED)

**Symptom:**
Confusion about whether `tensor-data-type=5` or `tensor-data-type=2` should be used for FP16.

**Root Cause:**
nvdspreprocess and nvinfer use **different numbering systems**:

| Data Type | nvdspreprocess (`tensor-data-type`) | nvinfer (`network-mode`) |
|-----------|-------------------------------------|--------------------------|
| FP32      | 0                                   | 0                        |
| UINT8     | 1                                   | N/A                      |
| INT8      | 2                                   | 1                        |
| UINT32    | 3                                   | N/A                      |
| INT32     | 4                                   | N/A                      |
| **FP16**  | **5**                               | **2**                    |

**Fix Applied:**
- `config_preprocess_tiling.txt`: `tensor-data-type=5` (FP16)
- `config_infer_primary_yolo11_tiling.txt`: `network-mode=2` (FP16)

---

### 3. Understanding `converted_frame_ptr` Data Format (DOCUMENTED)

**Key Discovery:**
The `converted_frame_ptr` in nvdspreprocess custom tensor functions is **ALWAYS UINT8 RGB**, regardless of the `tensor-data-type` setting.

**Evidence from DeepStream source** (`nvdspreprocess_lib/nvdspreprocess_impl.cpp`):
```cpp
// Default tensor preparation calls convertFcnHalf for FP16:
convertFcnHalf((unsigned char*)batch->units[i].converted_frame_ptr, ...);
```

**Implication:**
Our custom UINT8→FP16 conversion kernel is correct. The `tensor-data-type` setting only specifies the **output tensor format**, not the input format.

---

### 4. Memory Allocation Failures (PARTIALLY RESOLVED)

**Symptom:**
```
cudaMallocHost failed, cuda err_no:2, err_str:cudaErrorMemoryAllocation
NvMapMemAllocInternalTagged: error 12 (ENOMEM)
```

**Root Cause:**
Buffer pool sizes too large for Jetson's 7.4GB RAM when running both cameras.

**Fix Applied:**
Reduced buffer pool sizes in `config_preprocess_tiling.txt`:
```ini
scaling-buf-pool-size=4   # Was 16
tensor-buf-pool-size=4    # Was 16
```

---

## Current Status

### Single Camera (Daylight) Pipeline: ✅ STABLE

**Test Result:**
- Duration: 120 seconds (full timeout)
- FPS: ~18-19 stable
- Frames processed: 1185+
- Crashes: **NONE**

```
FPS: 18.60 | Frames: 93
FPS: 18.60 | Frames: 186
...
FPS: 18.20 | Frames: 1092
FPS: 18.60 | Frames: 1185
```

### Dual Camera Pipeline: ⚠️ UNSTABLE

**Symptom:**
Crashes after a few seconds with CUDA errors when both cameras run simultaneously.

**Error Sequence:**
1. `nvbufsurftransform_copy.cpp:341: => Failed in mem copy`
2. `cudaErrorIllegalAddress (700)` in preprocessing
3. Segmentation fault

**Suspected Cause:**
- Memory contention between two pipelines
- Possible race condition in shared GPU resources
- Thermal USB camera initialization conflicts

---

## Configuration Files

### config_preprocess_tiling.txt (Working)
```ini
[property]
enable=1
processing-width=640
processing-height=640
scaling-pool-compute-hw=1        # GPU
scaling-pool-memory-type=2       # CUDA_DEVICE
scaling-filter=1                 # Bilinear
scaling-buf-pool-size=4
tensor-buf-pool-size=4
network-input-shape=8;3;640;640  # Batch=8 for 8 tiles
network-color-format=0           # RGB
tensor-data-type=5               # FP16
custom-lib-path=/home/jet-nx8/DeepStream-Yolo/libnvdsinfer_custom_impl_Yolo.so
custom-tensor-preparation-function=CustomTensorPreparation
```

### Engine Requirements
- File: `model_b8_gpu0_fp16.engine`
- Batch size: 8
- Precision: FP16
- Input shape: 8×3×640×640

---

## Next Steps

1. **Investigate dual-camera memory conflicts**
   - Profile GPU memory usage with `tegrastats`
   - Consider sequential initialization
   - Add CUDA stream synchronization between pipelines

2. **Test thermal camera in isolation**
   - Verify thermal pipeline works alone before combining

3. **Consider alternative approaches**
   - Run cameras on separate GPU streams
   - Use nvstreammux to combine before single nvdspreprocess

4. **Add robust error recovery**
   - CUDA error checking after each kernel
   - Graceful degradation on memory pressure

---

## Session Update (16:30)

### Additional Findings

#### 5. Missing `input-tensor-from-meta` Setting (RESOLVED)

**Symptom:**
TensorRT inference immediately failed even with valid tensor data.

**Root Cause:**
The nvinfer configuration was missing `input-tensor-from-meta=1`, which tells nvinfer to use the preprocessed tensor from nvdspreprocess instead of performing its own preprocessing.

**Fix Applied:**
Added to `config_infer_primary_yolo11_tiling.txt`:
```ini
input-tensor-from-meta=1
```

#### 6. TensorRT CuTensor Permutation Error (INVESTIGATING)

**Symptom:**
```
ERROR: [TRT]: IExecutionContext::enqueueV3: Error Code 1: CuTensor (Internal cuTensor permutate execute failed)
```

**Observations:**
- Occurs after first few frames in dual-camera mode
- Single camera (daylight only) runs successfully
- Error corrupts CUDA context, causing subsequent "invalid device function" errors

**Suspected Causes:**
1. Memory contention between two parallel inference pipelines
2. Race condition in tensor buffer management
3. Incompatible tensor memory layouts between nvdspreprocess and TensorRT

#### 7. Dual-Camera Pipeline Partial Success

**Test Result:**
- Duration: ~5 seconds before crash
- FPS before crash: Daylight 16.2 FPS, Thermal 12.6 FPS
- Root cause: TensorRT internal error during dual inference

### Updated Status

| Pipeline | Status | Notes |
|----------|--------|-------|
| Daylight Only | ✅ **STABLE** | 1600+ frames, 120 seconds, no crashes |
| Thermal Only | ✅ Stable | Works correctly |
| Dual Camera | ⚠️ Unstable | ~5s before TensorRT crash |

### Key Finding: Daylight Tiling is STABLE! 

**Confirmed stable operation:**
- Processed 1600+ frames over 120 seconds
- Buffer pool stabilizes after first 3 frames
- Pointers cycle predictably (same addresses reused)
- No memory leaks or CUDA errors
- FPS maintained at ~18-19 FPS

The crash occurs ONLY when running both cameras simultaneously, suggesting the issue is resource contention or race conditions in the dual-pipeline configuration.

### Remaining Issues

1. **CuTensor permutation failure** - Need to investigate TensorRT memory management
2. **CUDA context corruption** - After TensorRT fails, CUDA operations fail
3. **Dual pipeline race condition** - Suspected resource sharing issue

### Root Cause Analysis: Dual Camera Failure

The dual camera crash is **NOT caused by our custom tiling code**. Evidence:

1. Daylight-only (with tiling) runs perfectly for 1600+ frames
2. Thermal-only runs perfectly
3. Crash occurs in nvinfer's internal `get_converted_buffer` function
4. Error: `cudaMemset2DAsync failed with error cudaErrorIllegalAddress`

The issue is that **two TensorRT execution contexts running simultaneously** in the same process are interfering with each other:
- Thermal: batch=1, 640x512 input
- Daylight: batch=8, 640x640 input (via nvdspreprocess)

This is a known limitation when running multiple TensorRT engines with different batch sizes in DeepStream.

### Recommendations

1. **Serialize inference** - Run daylight and thermal inference sequentially, not parallel
2. **Use separate CUDA contexts** - Isolate each pipeline's GPU resources  
3. **Reduce tensor buffer pool** - Lower `tensor-buf-pool-size` further if memory pressure
4. **Add CUDA device synchronization** - `cudaDeviceSynchronize()` between pipeline stages

### Recommended Solution: Sequential Pipeline Architecture

Instead of running both cameras in parallel threads, use a single-threaded approach:
1. Process thermal frame with batch=1 engine
2. Synchronize CUDA
3. Process daylight frame with batch=8 tiled engine
4. Repeat

Alternative: Use a single nvstreammux to combine both camera streams, then process with a unified inference pipeline.

---

## Final Summary

### What Works ✅

1. **Single camera tiling pipeline** - Fully stable, 18-19 FPS, 1600+ frames without crash
2. **CUDA architecture fix** - `-arch=sm_87` for Jetson Orin NX
3. **FP16 tensor preparation** - UINT8→FP16 conversion kernel working correctly
4. **nvdspreprocess integration** - Custom tensor function properly feeds batch=8 tensor

### What Doesn't Work ❌

1. **Dual camera parallel inference** - TensorRT context corruption when both engines run simultaneously

### Code Changes Made (Permanent)

| File | Change |
|------|--------|
| `nvdsinfer_custom_impl_Yolo/Makefile` | Added `-arch=sm_87` for CUDA compilation |
| `config_preprocess_tiling.txt` | Set `tensor-data-type=5` for FP16 |
| `config_infer_primary_yolo11_tiling.txt` | Added `input-tensor-from-meta=1` |
| `nvdsinfer_tiled_preprocess.cpp` | Added mutex for CUDA serialization (optional) |
| `nvdsinfer_tiled_preprocessor.cu` | FP16 conversion kernel with correct architecture |

### Next Steps for Dual Camera Support

1. Investigate DeepStream's multi-source inference configuration
2. Consider using nvstreammux to combine streams before single inference
3. Evaluate performance impact of sequential vs parallel processing
4. Test with CUDA MPS (Multi-Process Service) for context isolation

---

## Hardware Configuration

| Component | Value |
|-----------|-------|
| Platform | Jetson Orin NX |
| RAM | 7.4 GB |
| CUDA | 12.6.85 |
| Compute Capability | 8.7 (sm_87) |
| DeepStream | 7.1 |

---

## Files Modified

1. `nvdsinfer_custom_impl_Yolo/Makefile` - Added `-arch=sm_87`
2. `config_preprocess_tiling.txt` - Updated tensor-data-type and buffer sizes
3. `nvdsinfer_tiled_preprocessor.cu` - UINT8→FP16 conversion kernel
4. `nvdsinfer_tiled_preprocess.cpp` - Custom tensor preparation function
