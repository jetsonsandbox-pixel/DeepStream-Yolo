# C++ NMS Library for Tiled Detection

Fast C++ implementation of Non-Maximum Suppression for tiled object detection merging.

## Performance

- **Speedup**: ~4% FPS improvement (7.1 → 7.4 FPS)
- **Language**: C++11 with Python ctypes integration
- **Size**: 14KB compiled library

## Files

```
nms_cpp/
├── nms_merger.cpp      # C++ NMS implementation
├── Makefile            # Build configuration
└── libnms_merger.so    # Compiled library
```

## Building

```bash
cd nms_cpp
make clean && make
make install  # Copies to parent directory
```

## Usage

### Python Integration (Automatic)

The `DetectionMerger` class automatically uses C++ NMS when available:

```python
from tiled_yolo_inference import DetectionMerger

# Automatically loads C++ library
merger = DetectionMerger(use_cpp=True)  # Default

# Falls back to Python if C++ unavailable
merger = DetectionMerger(use_cpp=False)  # Force Python
```

### Direct ctypes Usage

```python
import ctypes
import numpy as np

# Load library
lib = ctypes.CDLL('libnms_merger.so')

# Setup function signatures
lib.nms_merge_detections.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # detections (N, 6) flattened
    ctypes.c_int,                     # num_detections
    ctypes.c_float,                   # nms_threshold
    ctypes.POINTER(ctypes.c_int),     # output_indices
    ctypes.POINTER(ctypes.c_int)      # num_kept
]
lib.nms_merge_detections.restype = ctypes.c_int

# Prepare data
detections = np.array([
    [100, 100, 200, 200, 0.9, 0],  # [x1, y1, x2, y2, conf, class_id]
    [110, 110, 210, 210, 0.8, 0],  # Overlapping box
    [300, 300, 400, 400, 0.7, 1],  # Different location
], dtype=np.float32)

num_det = len(detections)
output_indices = np.zeros(num_det, dtype=np.int32)
num_kept = ctypes.c_int(0)

# Call NMS
result = lib.nms_merge_detections(
    detections.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    num_det,
    ctypes.c_float(0.45),  # IoU threshold
    output_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.byref(num_kept)
)

# Get filtered detections
if result == 0:
    kept_detections = detections[output_indices[:num_kept.value]]
    print(f"Kept {num_kept.value} detections after NMS")
```

## API Reference

### `nms_merge_detections`

Apply Non-Maximum Suppression to detections.

**Signature:**
```c
int nms_merge_detections(
    const float* detections,      // Input: (N, 6) array [x1, y1, x2, y2, conf, class_id]
    int num_detections,            // Number of detections
    float nms_threshold,           // IoU threshold (0.0-1.0)
    int* output_indices,           // Output: indices of kept detections
    int* num_kept                  // Output: number of kept detections
)
```

**Returns:** 0 on success, -1 on error

**Parameters:**
- `detections`: Flattened array of shape (N, 6) where each detection is [x1, y1, x2, y2, confidence, class_id]
- `num_detections`: Number of input detections
- `nms_threshold`: IoU threshold for suppression (typical: 0.45)
- `output_indices`: Pre-allocated array (size N) to store indices of kept detections
- `num_kept`: Pointer to int that receives number of kept detections

### `nms_version`

Get library version string.

**Signature:**
```c
const char* nms_version()
```

**Returns:** Version string (e.g., "1.0.0")

## Algorithm

1. **Sort** detections by confidence (descending)
2. **Iterate** through sorted detections
3. **Keep** current detection
4. **Calculate** IoU with remaining detections of same class
5. **Suppress** detections with IoU > threshold
6. **Repeat** until all processed

## Performance Characteristics

- **Time Complexity**: O(N²) worst case, O(N) best case
- **Space Complexity**: O(N)
- **Optimizations**: 
  - Early termination for no overlap
  - Class-based filtering
  - Sorted confidence for better pruning

## Benchmark Results

Tested on Jetson Orin NX with 1920×1080 aerial footage:

| Implementation | FPS | Avg Time | Speedup |
|----------------|-----|----------|---------|
| Python NMS | 7.1 | 140.8ms | Baseline |
| **C++ NMS** | **7.4** | **135.1ms** | **+4%** |

## Troubleshooting

### Library not found
```bash
# Check if library exists
ls -lh libnms_merger.so

# Rebuild if needed
cd nms_cpp && make clean && make install
```

### Symbol errors
```bash
# Check exported symbols
nm -D libnms_merger.so | grep nms_merge

# Should show:
# T nms_merge_detections
# T nms_version
```

### Python import fails
```python
# Test loading
import ctypes
lib = ctypes.CDLL('./libnms_merger.so')
print(lib.nms_version().decode('utf-8'))  # Should print "1.0.0"
```

## Integration Notes

The library is automatically used by:
- `tiled_yolo_inference.py` → `DetectionMerger` class
- `realtime_tiled_detection.py` → via `TiledYOLOInference`
- `test_tiled_pipeline.py` → in all tests

Look for log message: **"Using C++ NMS library v1.0.0 ⚡"**

## Future Enhancements

- [ ] SIMD vectorization for IoU calculation
- [ ] Multi-threaded NMS for large detection sets
- [ ] GPU-accelerated NMS using CUDA
- [ ] Support for rotated bounding boxes

## License

Same as parent repository.

---

**Status**: ✅ Production-ready  
**Version**: 1.0.0  
**Speedup**: ~4% FPS improvement  
**Compiled**: November 2025
