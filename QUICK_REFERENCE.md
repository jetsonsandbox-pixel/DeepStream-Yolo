# Tiled Inference Quick Reference

## Quick Start Commands

### Process Video (With Display)
```bash
cd /home/jet-nx8/DeepStream-Yolo
source /home/jet-nx8/Sandbox/ultralytics-env/bin/activate
python3 realtime_tiled_detection.py --input video.mp4 --output result.mp4 --conf 0.25
```

### Process Video (Headless)
```bash
cd /home/jet-nx8/DeepStream-Yolo
source /home/jet-nx8/Sandbox/ultralytics-env/bin/activate
python3 realtime_tiled_detection.py --input video.mp4 --output result.mp4 --no-display --conf 0.25
```

### Process Camera Feed
```bash
python3 realtime_tiled_detection.py --input 0 --output camera.mp4 --conf 0.3
```

### Test Pipeline
```bash
python3 test_tiled_pipeline.py
```

## Performance Tips

### Clear Memory Before Running
```bash
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

### Check Memory Usage
```bash
free -h
tegrastats
```

### Monitor GPU
```bash
nvidia-smi
```

## File Locations

| File | Purpose |
|------|---------|
| `tiled_yolo_inference.py` | Core pipeline |
| `realtime_tiled_detection.py` | Video processor |
| `test_tiled_pipeline.py` | Tests |
| `model_b8_gpu0_fp32.engine` | TensorRT engine |
| `labels.txt` | Class names |
| `REALTIME_DETECTION.md` | Full docs |

## Performance Metrics

- **FPS**: 7.1 average (5.8-7.7 range)
- **Latency**: 140ms per frame
- **Input**: 1920×1080
- **Tiles**: 8 × 640×640 (96px overlap)
- **Memory**: ~5.8GB / 7.4GB

## Common Issues

### Out of Memory
```bash
# Clear cache
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Or reduce confidence threshold
python3 realtime_tiled_detection.py --input video.mp4 --conf 0.5
```

### No Detections
```bash
# Lower confidence
python3 realtime_tiled_detection.py --input video.mp4 --conf 0.1
```

### Low FPS
```bash
# Disable display
python3 realtime_tiled_detection.py --input video.mp4 --no-display

# Higher confidence
python3 realtime_tiled_detection.py --input video.mp4 --conf 0.5
```

## Python API

```python
from tiled_yolo_inference import TiledYOLOInference, TileConfig

# Initialize
config = TileConfig()  # 1920×1080 → 8 tiles
pipeline = TiledYOLOInference('model_b8_gpu0_fp32.engine', config)

# Process frame
import cv2
frame = cv2.imread('image.jpg')
detections = pipeline.process_frame(frame)  # (N, 6) [x1,y1,x2,y2,conf,class_id]
```

## Interactive Controls

- **`q`** - Quit
- **`s`** - Save screenshot

## Configuration

### Tile Configuration (in tiled_yolo_inference.py)
```python
config = TileConfig(
    frame_width=1920,
    frame_height=1080,
    tile_size=640,
    overlap=96
)
```

### Detection Parameters
```python
pipeline = TiledYOLOInference(
    engine_path='model_b8_gpu0_fp32.engine',
    config=config
)

# Parse with custom confidence
detections = pipeline.parse_yolo_output(output, conf_threshold=0.25)
```

## Git

**Branch**: `feature/tiling`

```bash
# Check current branch
git branch

# View changes
git status
git diff

# Commit changes
git add .
git commit -m "Tiled inference implementation"
```

## Documentation Files

- `REALTIME_DETECTION.md` - User guide with examples
- `TILED_INFERENCE_SUMMARY.md` - Implementation details
- `docs/PYCUDA_VS_PYTORCH_DECISION.md` - Technical decision doc
- `README.md` - Main repository readme (updated)

## Example Output

```
[VideoProcessor] Processing complete!
   Total frames: 100
   Total time: 14.2s
   Average FPS: 7.1
   FPS range: 5.8 - 7.7
   Output saved to: output_detections.mp4
```

## Need Help?

1. Read `REALTIME_DETECTION.md` for detailed guide
2. Check `TILED_INFERENCE_SUMMARY.md` for implementation details
3. Run tests: `python3 test_tiled_pipeline.py`
4. Check logs for error messages
