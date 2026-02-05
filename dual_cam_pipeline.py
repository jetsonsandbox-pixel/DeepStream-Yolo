#!/usr/bin/env python3
"""
Dual Camera DeepStream Pipeline
- Daylight CSI: 1920x1080 with tiling (8 tiles, batch=8)
- Thermal USB: 640x512 without tiling (batch=1)
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DualCameraPipeline:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("dual-camera-pipeline")
        self.loop = GLib.MainLoop()
        
        # FPS tracking
        self.daylight_frame_count = 0
        self.thermal_frame_count = 0
        self.daylight_start_time = None
        self.thermal_start_time = None
        self.last_fps_print = time.time()
        
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
        
        # Add queue after preprocessing to prevent buffer overflow
        preprocess_queue = Gst.ElementFactory.make("queue", "daylight-preprocess-queue")
        preprocess_queue.set_property("max-size-buffers", 4)
        preprocess_queue.set_property("max-size-bytes", 0)
        preprocess_queue.set_property("max-size-time", 0)
        preprocess_queue.set_property("leaky", 0)  # No leaky, block instead
        
        # Inference with daylight model
        infer = Gst.ElementFactory.make("nvinfer", "daylight-infer")
        infer.set_property("config-file-path", "config_infer_primary_yolo11_tiling.txt")
        infer.set_property("gpu-id", 0)
        infer.set_property("input-tensor-meta", True)
        
        # Add queue after inference
        infer_queue = Gst.ElementFactory.make("queue", "daylight-infer-queue")
        infer_queue.set_property("max-size-buffers", 2)
        infer_queue.set_property("max-size-bytes", 0)
        infer_queue.set_property("max-size-time", 0)
        infer_queue.set_property("leaky", 2)  # Leak downstream (drop old buffers)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "daylight-osd")
        
        # Add probe for FPS measurement
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.daylight_fps_probe, None)
        
        # Sink - use fakesink to avoid GPU display conflicts
        # Multiple sinks can't safely share GPU memory
        sink = Gst.ElementFactory.make("nveglglessink", "daylight-sink")
        sink.set_property("sync", False)
        
        # Add elements
        elements = [src, caps_filter, mux, preprocess, preprocess_queue, infer, infer_queue, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)
        
        # Link elements
        src.link(caps_filter)
        
        # Link caps to mux sink pad
        src_pad = caps_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        # Link rest of pipeline
        # Link rest of pipeline
        mux.link(preprocess)
        preprocess.link(preprocess_queue)
        preprocess_queue.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        return True
    
    def build_thermal_branch(self):
        """Build thermal USB branch without tiling"""
        # Source: USB camera (thermal)
        src = Gst.ElementFactory.make("v4l2src", "thermal-src")
        src.set_property("device", "/dev/video1")

        # Caps: native thermal resolution and format
        caps = Gst.Caps.from_string(
            "video/x-raw, width=640, height=512, format=YUY2, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "thermal-caps")
        caps_filter.set_property("caps", caps)

        # Video convert to match CUDA expectations
        convert = Gst.ElementFactory.make("videoconvert", "thermal-convert")

        # Queue to decouple from nvvideoconvert
        queue = Gst.ElementFactory.make("queue", "thermal-queue")
        queue.set_property("max-size-buffers", 3)

        # NVMM upload
        nvconvert = Gst.ElementFactory.make("nvvideoconvert", "thermal-nvconvert")
        nvconvert.set_property("nvbuf-memory-type", 0)

        # NVMM caps for TensorRT
        nvmm_caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12"
        )
        nvmm_filter = Gst.ElementFactory.make("capsfilter", "thermal-nvmm-caps")
        nvmm_filter.set_property("caps", nvmm_caps)
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "thermal-mux")
        mux.set_property("width", 640)
        mux.set_property("height", 512)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        mux.set_property("batched-push-timeout", 40000)
        
        # Inference with thermal model (no preprocessing)
        infer = Gst.ElementFactory.make("nvinfer", "thermal-infer")
        infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
        infer.set_property("gpu-id", 0)
        
        # Add queue after inference
        infer_queue = Gst.ElementFactory.make("queue", "thermal-infer-queue")
        infer_queue.set_property("max-size-buffers", 2)
        infer_queue.set_property("max-size-bytes", 0)
        infer_queue.set_property("max-size-time", 0)
        infer_queue.set_property("leaky", 2)  # Leak downstream (drop old buffers)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "thermal-osd")
        
        # Add probe for FPS measurement
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.thermal_fps_probe, None)
        
        # Sink - use fakesink to avoid GPU display conflicts
        # Multiple sinks can't safely share GPU memory
        sink = Gst.ElementFactory.make("nveglglessink", "thermal-sink")
        sink.set_property("sync", False)
        
        # Add elements
        elements = [src, caps_filter, convert, queue, nvconvert, nvmm_filter, 
               mux, infer, infer_queue, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)
        
        # Link elements
        src.link(caps_filter)
        caps_filter.link(convert)
        convert.link(queue)
        queue.link(nvconvert)
        nvconvert.link(nvmm_filter)
        
        # Link to mux sink pad
        src_pad = nvmm_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        # Link rest of pipeline
        mux.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        return True
    
    def daylight_fps_probe(self, pad, info, user_data):
        """Measure daylight branch FPS"""
        if self.daylight_start_time is None:
            self.daylight_start_time = time.time()
        
        self.daylight_frame_count += 1
        
        # Print FPS every 5 seconds
        current_time = time.time()
        if current_time - self.last_fps_print >= 5.0:
            self.print_fps()
            self.last_fps_print = current_time
        
        return Gst.PadProbeReturn.OK
    
    def thermal_fps_probe(self, pad, info, user_data):
        """Measure thermal branch FPS"""
        if self.thermal_start_time is None:
            self.thermal_start_time = time.time()
        
        self.thermal_frame_count += 1
        return Gst.PadProbeReturn.OK
    
    def print_fps(self):
        """Print FPS for both branches"""
        current_time = time.time()
        
        if self.daylight_start_time:
            daylight_elapsed = current_time - self.daylight_start_time
            daylight_fps = self.daylight_frame_count / daylight_elapsed if daylight_elapsed > 0 else 0
        else:
            daylight_fps = 0
        
        if self.thermal_start_time:
            thermal_elapsed = current_time - self.thermal_start_time
            thermal_fps = self.thermal_frame_count / thermal_elapsed if thermal_elapsed > 0 else 0
        else:
            thermal_fps = 0
        
        print(f"\rDaylight: {daylight_fps:.1f} FPS | Thermal: {thermal_fps:.1f} FPS", end="", flush=True)
    
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
