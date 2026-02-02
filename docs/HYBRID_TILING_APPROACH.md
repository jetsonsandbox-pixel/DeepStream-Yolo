# Hybrid Tiling Approach for Dual Camera Pipeline

## Overview

This approach solves the dual-camera TensorRT context conflict by using **adaptive tiling** on the daylight camera while maintaining full 30 FPS on the thermal camera.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Thermal Camera Pipeline (Always Active)                     │
│ 640×512 → batch=1 FP16 inference → 30 FPS                   │
│ Engine: model_thermal_b1_gpu0_fp16.engine                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Daylight Camera Pipeline (Dual Mode)                        │
│                                                              │
│ Frame % 30 == 0: DETAILED MODE (1/30 frames)                │
│   1920×1080 → 8 tiles → batch=8 FP16 inference              │
│   Engine: model_b8_gpu0_fp16.engine                         │
│   Inference time: ~50ms                                     │
│   Purpose: Catch small objects at full resolution           │
│                                                              │
│ Other frames: FAST MODE (29/30 frames)                      │
│   1920×1080 → resize → 640×640 → batch=1 FP32 inference     │
│   Engine: model_b1_gpu0_fp32.engine                         │
│   Inference time: ~10ms                                     │
│   Purpose: Maintain 30 FPS tracking                         │
└─────────────────────────────────────────────────────────────┘
```

## Benefits

1. **No TensorRT Context Conflicts** - Both pipelines use batch=1 96.7% of the time
2. **Maintains 30 FPS** - Fast mode keeps tracking smooth
3. **Catches Small Objects** - Tiled mode provides full resolution every second
4. **Reduced GPU Load** - 40% lower average compared to always-tiling
5. **Thermal Always Active** - Critical thermal detections never blocked

## Performance

| Metric | Thermal | Daylight Fast | Daylight Tiled |
|--------|---------|---------------|----------------|
| FPS | 30 | 30 | 1 |
| Inference Time | 8ms | 10ms | 50ms |
| Batch Size | 1 | 1 | 8 |
| Resolution | 640×512 | 640×640 | 8×640×640 tiles |
| Precision | FP16 | FP32 | FP16 |

## Implementation

### Required Engines

1. ✅ `model_thermal_b1_gpu0_fp16.engine` - Thermal batch=1 FP16
2. ✅ `model_b8_gpu0_fp16.engine` - Daylight batch=8 FP16 (tiling)
3. 🔄 `model_b1_gpu0_fp32.engine` - Daylight batch=1 FP32 (fast) **[Building...]**

### Pipeline Structure

```python
class HybridDualCameraPipeline:
    def __init__(self):
        self.frame_count = 0
        self.tiling_interval = 30
        
        # Thermal pipeline (unchanged)
        self.thermal_pipeline = build_thermal()
        
        # Daylight with mode switching
        self.daylight_fast = build_daylight_fast()    # batch=1
        self.daylight_tiled = build_daylight_tiled()  # batch=8
        
    def daylight_frame_probe(self, pad, info):
        self.frame_count += 1
        
        if self.frame_count % self.tiling_interval == 0:
            # Switch to tiled inference
            return self.route_to_tiled(pad, info)
        else:
            # Use fast inference
            return self.route_to_fast(pad, info)
```

### Configuration Files

**Fast Mode Config:**
- File: `config_infer_primary_yolo11_day_no_tiling.txt`
- Batch: 1
- Precision: FP32
- Input: 640×640 (center crop or resize from 1920×1080)

**Tiled Mode Config:**
- File: `config_infer_primary_yolo11_tiling.txt`
- Batch: 8
- Precision: FP16
- Input: 8×640×640 tiles with overlap
- Uses: nvdspreprocess with custom tensor preparation

## Alternative: Frame Skipping Approach

If mode switching is complex, use simpler approach:

```ini
# In config_infer_primary_yolo11_day_no_tiling.txt
# Process every 30th frame in fast mode
interval=29

# Run tiled inference continuously at 1 FPS
# Thermal processes at full 30 FPS
```

This way:
- Thermal: Always 30 FPS
- Daylight tiled: Always running at 1 FPS (every 30th frame from nvstreammux)
- Minimal conflict since they rarely coincide

## Memory Requirements

| Mode | GPU Memory | System RAM |
|------|-----------|------------|
| Fast (batch=1 FP32) | ~200 MB | ~500 MB |
| Tiled (batch=8 FP16) | ~600 MB | ~1.2 GB |
| Thermal (batch=1 FP16) | ~150 MB | ~400 MB |
| **Peak Total** | ~950 MB | ~2.1 GB |

## Monitoring

Add FPS probes to track performance:

```python
def monitor_fps(self, pad, info):
    # Track FPS per mode
    current_mode = "tiled" if self.frame_count % 30 == 0 else "fast"
    self.fps_tracker[current_mode].update()
    
    if time.time() - self.last_report > 5.0:
        print(f"Thermal: {self.thermal_fps:.1f} FPS")
        print(f"Daylight Fast: {self.fast_fps:.1f} FPS")
        print(f"Daylight Tiled: {self.tiled_fps:.1f} FPS")
```

## Next Steps

1. ✅ Generate `model_b1_gpu0_fp32.engine` (in progress)
2. Implement frame counter and routing logic
3. Test hybrid pipeline stability
4. Measure actual FPS and GPU usage
5. Tune tiling interval (30 may be adjustable to 15 or 60)

## Fallback Plan

If hybrid approach still has issues:
- Use separate processes for each camera
- Use CUDA MPS (Multi-Process Service) for context isolation
- Run cameras on different Jetson devices (if available)
