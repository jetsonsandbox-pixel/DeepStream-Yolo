# Dual Camera DeepStream Pipeline Implementation Plan

## Overview

Deploy two YOLO11n models on Jetson Orin NX for simultaneous daylight and thermal vision detection with two display outputs.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BRANCH A: DAYLIGHT CSI                      │
├─────────────────────────────────────────────────────────────────┤
│  nvarguscamerasrc (CSI, 1920x1080@30)                           │
│           ↓                                                      │
│  nvstreammux (batch=1, 1920x1080)                               │
│           ↓                                                      │
│  nvdspreprocess (custom tiler: 8 tiles 640x640, overlap=96px)  │
│           ↓                                                      │
│  nvinfer (YOLO11n daylight, batch=8, gie-id=1)                 │
│           ↓                                                      │
│  nvdsosd (draw detections)                                      │
│           ↓                                                      │
│  nveglglessink (Window 1: Daylight)                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      BRANCH B: THERMAL USB                       │
├─────────────────────────────────────────────────────────────────┤
│  v4l2src (USB, 640x512, /dev/video1)                           │
│           ↓                                                      │
│  nvstreammux (batch=1, 640x512)                                 │
│           ↓                                                      │
│  nvinfer (YOLO11n thermal, batch=1, gie-id=2, no tiling)       │
│           ↓                                                      │
│  nvdsosd (draw detections)                                      │
│           ↓                                                      │
│  nveglglessink (Window 2: Thermal)                              │
└─────────────────────────────────────────────────────────────────┘
```

## Current Status

### Working Components
- **Daylight CSI**: 1920x1080@30 via CSI camera
- **Daylight Preprocessing**: Custom CUDA tiler (8 tiles, 640x640, 96px overlap)
- **Daylight Model**: `yolo11n_daylight_2026-01-16_v4-8a.pt.onnx`
- **Daylight Engine**: `model_b8_gpu0_fp16.engine` (batch=8)
- **Performance**: ~25 FPS with tiling
- **Custom Library**: `libnvdsinfer_custom_impl_Yolo.so` (built, exports CustomTensorPreparation)

### To Be Implemented
- **Thermal USB Camera**: 640x512 via V4L2 USB
- **Thermal Model Export**: ONNX from `yolo11n_thermal_2026-01-09_v1-9c.pt`
- **Thermal Engine**: TensorRT FP16 (batch=1)
- **Thermal Config**: `config_infer_primary_thermal.txt`
- **Python Launcher**: `dual_cam_pipeline.py`
- **Labels**: `labels_thermal.txt`

## Implementation Steps

### Step 1: Export Thermal Model to ONNX

```bash
cd /home/jet-nx8/DeepStream-Yolo
source /home/jet-nx8/Sandbox/ultralytics-env/bin/activate

# Export thermal model
python utils/export_yolo11.py \
  --weights models-custom/yolo11n_thermal_2026-01-09_v1-9c.pt \
  --imgsz 640 512 \
  --batch 1 \
  --simplify

# Output: models-custom/yolo11n_thermal_2026-01-09_v1-9c.onnx
```

### Step 2: Create Thermal Inference Config

File: `config_infer_primary_thermal.txt`

```ini
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file=models-custom/yolo11n_thermal_2026-01-09_v1-9c.onnx
maintain-aspect-ratio=1
symmetric-padding=1

# Thermal camera: single frame processing (no tiling)
model-engine-file=model_thermal_b1_gpu0_fp16.engine
labelfile-path=labels_thermal.txt

# Single frame inference
batch-size=1

network-mode=2
num-detected-classes=8
interval=0

gie-unique-id=2
process-mode=1
network-type=0
cluster-mode=2
workspace-size=2000

parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so
engine-create-func-name=NvDsInferYoloCudaEngineGet

[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

### Step 3: Create Thermal Labels File

File: `labels_thermal.txt`

```
plane
missile
bird
unrecognized
drone
```

### Step 4: Create Dual Camera Pipeline Launcher

File: `dual_cam_pipeline.py`

```python
#!/usr/bin/env python3
"""
Dual Camera DeepStream Pipeline
- Daylight CSI: 1920x1080 with tiling (8 tiles, batch=8)
- Thermal USB: 640x512 without tiling (batch=1)
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DualCameraPipeline:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("dual-camera-pipeline")
        self.loop = GLib.MainLoop()
        
    def build_daylight_branch(self):
        """Build daylight CSI branch with tiling"""
        # Source: CSI camera
        src = Gst.ElementFactory.make("nvarguscamerasrc", "daylight-src")
        src.set_property("sensor-id", 0)
        
        # Caps: 1920x1080@30
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), width=1920, height=1080, "
            "format=NV12, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "daylight-caps")
        caps_filter.set_property("caps", caps)
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "daylight-mux")
        mux.set_property("width", 1920)
        mux.set_property("height", 1080)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        
        # Preprocessing with custom tiler
        preprocess = Gst.ElementFactory.make("nvdspreprocess", "daylight-preprocess")
        preprocess.set_property("config-file", "config_preprocess_tiling.txt")
        
        # Inference with daylight model
        infer = Gst.ElementFactory.make("nvinfer", "daylight-infer")
        infer.set_property("config-file-path", "config_infer_primary_yolo11_tiling.txt")
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "daylight-osd")
        
        # Sink
        sink = Gst.ElementFactory.make("nveglglessink", "daylight-sink")
        sink.set_property("sync", False)
        
        # Add elements
        elements = [src, caps_filter, mux, preprocess, infer, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)
        
        # Link elements
        src.link(caps_filter)
        
        # Link caps to mux sink pad
        src_pad = caps_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        # Link rest of pipeline
        mux.link(preprocess)
        preprocess.link(infer)
        infer.link(osd)
        osd.link(sink)
        
        return True
    
    def build_thermal_branch(self):
        """Build thermal USB branch without tiling"""
        # Source: USB camera
        src = Gst.ElementFactory.make("v4l2src", "thermal-src")
        src.set_property("device", "/dev/video1")
        
        # Caps: 640x512
        caps = Gst.Caps.from_string(
            "video/x-raw, width=640, height=512, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "thermal-caps")
        caps_filter.set_property("caps", caps)
        
        # Video convert
        convert = Gst.ElementFactory.make("videoconvert", "thermal-convert")
        
        # NVMM upload
        nvconvert = Gst.ElementFactory.make("nvvideoconvert", "thermal-nvconvert")
        
        # NVMM caps
        nvmm_caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"
        )
        nvmm_filter = Gst.ElementFactory.make("capsfilter", "thermal-nvmm-caps")
        nvmm_filter.set_property("caps", nvmm_caps)
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "thermal-mux")
        mux.set_property("width", 640)
        mux.set_property("height", 512)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        
        # Inference with thermal model (no preprocessing)
        infer = Gst.ElementFactory.make("nvinfer", "thermal-infer")
        infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "thermal-osd")
        
        # Sink
        sink = Gst.ElementFactory.make("nveglglessink", "thermal-sink")
        sink.set_property("sync", False)
        
        # Add elements
        elements = [src, caps_filter, convert, nvconvert, nvmm_filter, 
                   mux, infer, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)
        
        # Link elements
        src.link(caps_filter)
        caps_filter.link(convert)
        convert.link(nvconvert)
        nvconvert.link(nvmm_filter)
        
        # Link to mux sink pad
        src_pad = nvmm_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        # Link rest of pipeline
        mux.link(infer)
        infer.link(osd)
        osd.link(sink)
        
        return True
    
    def bus_call(self, bus, message, loop):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True
    
    def run(self):
        """Build and run pipeline"""
        print("Building daylight branch...")
        if not self.build_daylight_branch():
            print("Failed to build daylight branch")
            return False
        
        print("Building thermal branch...")
        if not self.build_thermal_branch():
            print("Failed to build thermal branch")
            return False
        
        # Setup bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        # Start playing
        print("Starting pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        # Cleanup
        self.pipeline.set_state(Gst.State.NULL)
        return True

if __name__ == "__main__":
    pipeline = DualCameraPipeline()
    sys.exit(0 if pipeline.run() else 1)
```

### Step 5: Verify USB Thermal Camera

```bash
# List video devices
ls -l /dev/video*

# Check thermal camera capabilities
v4l2-ctl --device=/dev/video1 --all

# Test thermal camera feed
gst-launch-1.0 v4l2src device=/dev/video1 ! \
  videoconvert ! autovideosink
```

## Configuration Summary

### Daylight Branch
- **Source**: CSI camera (`csi://0`)
- **Resolution**: 1920x1080@30
- **Preprocessing**: Custom CUDA tiler (8 tiles, 640x640, overlap=96px)
- **Config**: `config_preprocess_tiling.txt`
- **Model**: `yolo11n_daylight_2026-01-16_v4-8a.pt.onnx`
- **Engine**: `model_b8_gpu0_fp16.engine` (batch=8)
- **Inference Config**: `config_infer_primary_yolo11_tiling.txt`
- **GIE ID**: 1
- **Labels**: `labels.txt`
- **Performance**: ~25 FPS

### Thermal Branch
- **Source**: USB camera (`/dev/video1`)
- **Resolution**: 640x512@30
- **Preprocessing**: None (direct inference)
- **Model**: `yolo11n_thermal_2026-01-09_v1-9c.onnx`
- **Engine**: `model_thermal_b1_gpu0_fp16.engine` (batch=1)
- **Inference Config**: `config_infer_primary_thermal.txt`
- **GIE ID**: 2
- **Labels**: `labels_thermal.txt`
- **Expected Performance**: >30 FPS

## Resource Allocation

### GPU Memory
- Daylight branch: ~4GB (tiling batch=8)
- Thermal branch: ~1.5GB (batch=1)
- Total: ~5.5GB (within 7.4GB limit)

### CPU Load
- Daylight: Moderate (CUDA preprocessing)
- Thermal: Low (single frame)
- Combined: Should remain <80%

## Testing Procedure

### 1. Test Thermal Model Export
```bash
python utils/export_yolo11.py --weights models-custom/yolo11n_thermal_2026-01-09_v1-9c.pt
```

### 2. Test Thermal Camera
```bash
v4l2-ctl --device=/dev/video1 --all
gst-launch-1.0 v4l2src device=/dev/video1 ! videoconvert ! autovideosink
```

### 3. Test Thermal Branch Alone
```bash
# Modify dual_cam_pipeline.py to only build thermal branch
python dual_cam_pipeline.py
```

### 4. Test Full Dual Pipeline
```bash
python dual_cam_pipeline.py
```

### 5. Monitor Performance
```bash
# In separate terminal
tegrastats
# or
jtop
```

## Expected Results

### Display Output
- **Window 1**: Daylight CSI (1920x1080) with bounding boxes
- **Window 2**: Thermal USB (640x512) with bounding boxes

### Performance Metrics
- Daylight FPS: ~25 (maintained from single-camera)
- Thermal FPS: >30 (lightweight single-frame)
- Combined GPU load: 60-80%
- Memory usage: ~5.5GB

### Detection Quality
- Daylight: High-resolution objects with tiling
- Thermal: Heat signatures and infrared objects
- No interference between branches

## Troubleshooting

### Thermal Camera Not Found
```bash
# Check USB connection
lsusb
# Check video devices
ls -l /dev/video*
# Verify permissions
sudo usermod -a -G video $USER
```

### Engine Build Failure
```bash
# Check ONNX model
ls -lh models-custom/yolo11n_thermal_2026-01-09_v1-9c.onnx
# Verify TensorRT
/usr/src/tensorrt/bin/trtexec --help
```

### Out of Memory
- Reduce daylight overlap (96 → 64px)
- Use FP16 for both engines
- Enable frame skipping (`interval=5`)

### Pipeline Link Errors
- Check element availability: `gst-inspect-1.0 nvinfer`
- Verify config file paths are correct
- Ensure custom library is built

## Next Steps

1. ✅ Document plan
2. ⬜ Export thermal model to ONNX
3. ⬜ Create `config_infer_primary_thermal.txt`
4. ⬜ Create `labels_thermal.txt`
5. ⬜ Create `dual_cam_pipeline.py`
6. ⬜ Test thermal camera connection
7. ⬜ Test thermal branch alone
8. ⬜ Test full dual pipeline
9. ⬜ Tune performance parameters
10. ⬜ Deploy to production

---

*Implementation Plan Created: January 28, 2026*
