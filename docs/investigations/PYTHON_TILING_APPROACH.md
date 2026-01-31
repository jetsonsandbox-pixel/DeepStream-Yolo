# Alternative Approach: Python-based Tile Extraction
#
# Instead of complex CUDA kernels or uncertain ROI support, 
# we can extract tiles in Python and feed them as separate streams

## Why This Approach?

1. **Memory Safety**: Python manages memory, no CUDA bugs
2. **Simplicity**: Clear, debuggable tile extraction logic
3. **Control**: Full control over overlap, padding, etc.
4. **Stability**: No driver crashes from buggy kernels

## Implementation Strategy

### Option A: Python → Multiple GStreamer Streams

```python
# Extract 8 tiles from 1920×1080 frame in Python
tiles = extract_tiles(frame_1920x1080, grid=(4,2), size=640, overlap=96)

# Feed each tile as separate stream to nvstreammux
for i, tile in enumerate(tiles):
    appsrc_tile[i].emit("push-buffer", tile_to_gst_buffer(tile))

# nvstreammux batches 8 tiles
# nvinfer processes batch of 8
```

**Pros:**
- No CUDA kernel needed
- Easy to debug
- Very stable
- Can add preprocessing (enhancement, normalization)

**Cons:**
- Python tile extraction on CPU (~5-10ms per frame)
- Extra memcpy from CPU → GPU
- More complex pipeline setup

### Option B: appsrc → Single Stream with Manual Batching

```python
# Extract 8 tiles
tiles = extract_tiles(frame, grid=(4,2))

# Stack into single tensor (8, 3, 640, 640)
batch_tensor = np.stack(tiles)

# Push to appsrc as single batch
appsrc.emit("push-buffer", tensor_to_buffer(batch_tensor))

# nvinfer processes pre-batched input
```

**Pros:**
- Direct batch input
- Simpler than 8 streams
- No nvstreammux needed

**Cons:**
- Need to handle NvDsMeta manually
- Must map detections back to tiles

### Option C: GStreamer videoscale + roi

Use GStreamer's built-in videocrop and videoscale:

```
nvarguscamerasrc
  ├─> videocrop (tile 0) → videoscale → nvstreammux.sink_0
  ├─> videocrop (tile 1) → videoscale → nvstreammux.sink_1
  ├─> videocrop (tile 2) → videoscale → nvstreammux.sink_2
  ...
  └─> videocrop (tile 7) → videoscale → nvstreammux.sink_7
```

**Pros:**
- No custom code
- All GStreamer native
- GPU-accelerated (videoscale uses VIC)

**Cons:**
- Complex pipeline with 8 branches
- Higher memory usage (8 parallel streams)
- Harder to manage

## Recommended: Option A (Python Tiles → GStreamer)

### Implementation

```python
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class PythonTilingPipeline:
    def __init__(self):
        # Create appsrc for each tile
        self.tile_sources = []
        for i in range(8):
            src = Gst.ElementFactory.make("appsrc", f"tile-src-{i}")
            src.set_property("format", Gst.Format.TIME)
            src.set_property("is-live", True)
            caps = Gst.Caps.from_string(
                "video/x-raw,format=RGB,width=640,height=640,framerate=30/1"
            )
            src.set_property("caps", caps)
            self.tile_sources.append(src)
            
        # nvstreammux batches 8 tiles
        self.mux = Gst.ElementFactory.make("nvstreammux")
        self.mux.set_property("batch-size", 8)
        self.mux.set_property("width", 640)
        self.mux.set_property("height", 640)
        
        # nvinfer with batch=8 model
        self.infer = Gst.ElementFactory.make("nvinfer")
        self.infer.set_property("config-file-path", 
                                "config_infer_primary_yolo11_tiling.txt")
    
    def extract_tiles(self, frame_bgr):
        """Extract 8 tiles from 1920×1080 frame"""
        h, w = frame_bgr.shape[:2]
        tiles = []
        
        # 4×2 grid with 96px overlap
        stride = 544  # 640 - 96
        for row in range(2):
            for col in range(4):
                x = col * stride
                y = row * stride
                
                # Extract tile
                tile = frame_bgr[y:y+640, x:x+640]
                
                # Pad if needed (right/bottom edges)
                if tile.shape[0] < 640 or tile.shape[1] < 640:
                    tile = cv2.copyMakeBorder(
                        tile,
                        0, 640 - tile.shape[0],
                        0, 640 - tile.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)
                    )
                
                # Convert BGR → RGB
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tiles.append(tile_rgb)
        
        return tiles
    
    def process_frame(self, frame_bgr):
        """Extract tiles and push to GStreamer"""
        tiles = self.extract_tiles(frame_bgr)
        
        for i, tile in enumerate(tiles):
            # Create GStreamer buffer
            data = tile.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            
            # Set timestamp
            buf.pts = self.frame_count * (Gst.SECOND // 30)
            buf.duration = Gst.SECOND // 30
            
            # Push to appsrc
            ret = self.tile_sources[i].emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print(f"Error pushing tile {i}")
        
        self.frame_count += 1
```

### Performance Estimate

- **Tile extraction** (Python + OpenCV): ~5-10ms per frame
- **memcpy to GPU**: ~2-3ms
- **Total overhead**: ~10-15ms per frame
- **Expected FPS**: ~20-25 (vs 15 with CUDA kernel)

### Advantages Over CUDA Kernel

| Aspect | CUDA Kernel | Python Tiles |
|--------|-------------|--------------|
| **Stability** | ❌ Crashes | ✅ Very stable |
| **Debugging** | ❌ Hard | ✅ Easy |
| **Memory Safety** | ❌ Bugs | ✅ Safe |
| **Performance** | ~15 FPS | ~20-25 FPS |
| **Development** | ❌ Complex | ✅ Simple |
| **Maintenance** | ❌ C++/CUDA | ✅ Python |

## Next Steps

1. Implement Python tile extraction
2. Test with single camera
3. Benchmark performance vs CUDA
4. If stable for 10+ minutes, integrate thermal camera
5. Deploy to production

## Fallback: Full-Frame Processing

If tiling proves too complex:
- Process full 1920×1080 frame
- Accept lower small-object detection
- Much simpler and more stable
- Train model on full-res images

