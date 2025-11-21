# Real-Time Tiled Object Detection

A production-ready script for real-time object detection using tiled inference on Jetson platforms.

## Features

✅ **Real-time processing** at 7+ FPS (1920×1080 video with 8 tiles)  
✅ **Batch inference** - processes all 8 tiles in single TensorRT call  
✅ **Bounding box visualization** with class labels and confidence scores  
✅ **Performance overlay** showing FPS, detection count, and frame progress  
✅ **Video output** - save processed video with detections  
✅ **Headless mode** - run without display for server deployments  
✅ **Camera support** - works with USB cameras and CSI cameras  

## Quick Start

### Process Video File

```bash
# With display (X11 required)
python3 realtime_tiled_detection.py \
    --input /path/to/video.mp4 \
    --output detections_output.mp4 \
    --conf 0.25

# Headless mode (no display)
python3 realtime_tiled_detection.py \
    --input /path/to/video.mp4 \
    --output detections_output.mp4 \
    --no-display \
    --conf 0.25
```

### Process from Camera

```bash
# USB camera (index 0)
python3 realtime_tiled_detection.py \
    --input 0 \
    --output camera_detections.mp4 \
    --conf 0.3

# CSI camera (index 1)
python3 realtime_tiled_detection.py \
    --input 1 \
    --output csi_detections.mp4
```

### Process Limited Frames (Testing)

```bash
# Process only first 100 frames
python3 realtime_tiled_detection.py \
    --input video.mp4 \
    --output test_output.mp4 \
    --no-display \
    --max-frames 100
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--engine` | Path to TensorRT engine file | `model_b8_gpu0_fp32.engine` |
| `--input` | Input video file or camera index | **Required** |
| `--output` | Output video file path | None (no output) |
| `--no-display` | Disable display window | False |
| `--conf` | Detection confidence threshold | 0.25 |
| `--labels` | Path to class labels file | `labels.txt` |
| `--max-frames` | Maximum frames to process | None (all frames) |

## Interactive Controls (Display Mode)

- **`q`** - Quit processing
- **`s`** - Save screenshot of current frame

## Performance

Tested on **Jetson Orin NX** with YOLO11n model:

| Configuration | FPS | Processing Time | Memory Usage |
|--------------|-----|-----------------|--------------|
| 1920×1080 input | 7.1 | 140ms/frame | ~5.8GB |
| 8 tiles (640×640) | - | Single batch inference | - |
| Batch size = 8 | - | 4× faster than sequential | - |

### Performance Tips

1. **Lower confidence threshold** = more detections, slower NMS
2. **Larger input resolution** = more tiles, slower processing
3. **Disable display** (`--no-display`) saves ~5-10% overhead
4. **Use MP4 output** instead of AVI for smaller file sizes

## Output Video Format

- **Codec**: MP4V (compatible with most players)
- **Resolution**: Same as input video
- **Frame rate**: Same as input video
- **Features**:
  - Bounding boxes with class labels
  - Confidence scores
  - Real-time performance stats overlay

## Memory Management

If you encounter **Out of Memory (OOM)** errors:

```bash
# Clear system cache before running
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Then run detection
python3 realtime_tiled_detection.py --input video.mp4 --output out.mp4 --no-display
```

For low-memory systems, consider:
- Rebuilding TensorRT engine with batch=1
- Reducing input resolution
- Processing fewer tiles

## Architecture

```
Video Input (1920×1080)
    ↓
Tile Extraction (8 tiles: 640×640)
    ↓
Batch Preprocessing (normalize, HWC→CHW)
    ↓
TensorRT Inference (single batch call)
    ↓
YOLO Output Parsing (filter by confidence)
    ↓
NMS Merging (handle overlapping detections)
    ↓
Visualization (draw boxes + labels)
    ↓
Video Output / Display
```

## Example Use Cases

### 1. Aerial Surveillance
```bash
python3 realtime_tiled_detection.py \
    --input drone_footage.mp4 \
    --output surveillance_output.mp4 \
    --no-display \
    --conf 0.3
```

### 2. Traffic Monitoring
```bash
python3 realtime_tiled_detection.py \
    --input /dev/video0 \
    --output traffic_feed.mp4 \
    --conf 0.5
```

### 3. Wildlife Detection
```bash
python3 realtime_tiled_detection.py \
    --input wildlife_cam.mp4 \
    --output animals_detected.mp4 \
    --labels wildlife_labels.txt \
    --conf 0.2
```

## Troubleshooting

### Video won't open
- Check file path is correct
- Verify codec support: `ffmpeg -i video.mp4`
- Try different video format (MP4, AVI, MOV)

### Low FPS
- Disable display: `--no-display`
- Increase confidence threshold: `--conf 0.5`
- Check CPU/GPU usage: `tegrastats`
- Close other GPU applications

### No detections
- Lower confidence: `--conf 0.1`
- Check if objects are in COCO classes (see `labels.txt`)
- Verify model is trained on target objects

### Display window issues
- Ensure X11 is available: `echo $DISPLAY`
- Use VNC or SSH with X forwarding
- Or use `--no-display` for headless operation

## Integration Examples

### Python Script Integration

```python
from realtime_tiled_detection import VideoProcessor

# Create processor
processor = VideoProcessor(
    engine_path='model_b8_gpu0_fp32.engine',
    input_source='video.mp4',
    output_path='output.mp4',
    display=False,
    conf_threshold=0.25
)

# Process video
processor.process_stream(max_frames=None)  # Process all frames
```

### Custom Detection Callback

```python
from tiled_yolo_inference import TiledYOLOInference, TileConfig
import cv2

# Initialize pipeline
config = TileConfig()
pipeline = TiledYOLOInference('model_b8_gpu0_fp32.engine', config)

# Process frames
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    detections = pipeline.process_frame(frame)
    
    # Custom handling
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        print(f"Detected class {int(class_id)} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] with {conf:.2f} confidence")
```

## Files

- `realtime_tiled_detection.py` - Main script for video processing
- `tiled_yolo_inference.py` - Core tiled inference pipeline
- `test_tiled_pipeline.py` - Unit and integration tests
- `labels.txt` - COCO class labels
- `model_b8_gpu0_fp32.engine` - TensorRT engine (batch=8)

## Requirements

- **Python 3.10+**
- **PyTorch 2.5.0** with CUDA support
- **TensorRT 10.7.0** with Python bindings
- **OpenCV 4.8+** with video codec support
- **NumPy 1.23+**

Install dependencies:
```bash
pip install opencv-python numpy torch
```

## License

See parent repository LICENSE.md

## Credits

Built on:
- NVIDIA DeepStream SDK
- Ultralytics YOLO
- TensorRT inference engine
- Tiled inference architecture from umbrella-jetson-dev

---

**Performance Tip**: For maximum FPS, use `--no-display --conf 0.5` to minimize overhead and filter weak detections.
