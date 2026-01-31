#!/usr/bin/env python3
"""Test dual camera pipeline with NO TILING - just to verify CUDA is stable without our kernel"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time

class DualCamNoTiling:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("dual-cam-no-tiling")
        self.loop = GLib.MainLoop()
        self.daylight_fps = 0
        self.thermal_fps = 0
        self.daylight_frame_count = 0
        self.thermal_frame_count = 0
        self.start_time = time.time()
        
    def build_daylight_branch(self):
        """Daylight camera branch - NO TILING, single frame batch=1"""
        print("Building daylight branch (no tiling)...")
        
        # CSI camera source
        src = Gst.ElementFactory.make("nvarguscamerasrc", "daylight-src")
        src.set_property("sensor-id", 0)
        
        # Camera capabilities
        caps_filter = Gst.ElementFactory.make("capsfilter", "daylight-caps")
        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1")
        caps_filter.set_property("caps", caps)
        
        # Queue
        queue = Gst.ElementFactory.make("queue", "daylight-queue")
        queue.set_property("max-size-buffers", 2)
        queue.set_property("leaky", 2)
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "daylight-mux")
        mux.set_property("width", 640)
        mux.set_property("height", 640)
        mux.set_property("batch-size", 1)  # Single frame, no tiling
        mux.set_property("live-source", True)
        mux.set_property("batched-push-timeout", 40000)
        
        # Inference - using batch=1 model (thermal model)
        infer = Gst.ElementFactory.make("nvinfer", "daylight-infer")
        infer.set_property("config-file-path", "config_infer_primary_daylight_no_tiling.txt")
        infer.set_property("gpu-id", 0)
        
        # Queue after inference
        infer_queue = Gst.ElementFactory.make("queue", "daylight-infer-queue")
        infer_queue.set_property("max-size-buffers", 2)
        infer_queue.set_property("leaky", 2)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "daylight-osd")
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.daylight_fps_probe, None)
        
        # Fakesink
        sink = Gst.ElementFactory.make("fakesink", "daylight-sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        
        # Add and link
        elements = [src, caps_filter, queue, mux, infer, infer_queue, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)
        
        src.link(caps_filter)
        caps_filter.link(queue)
        
        src_pad = queue.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        mux.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        return True
    
    def build_thermal_branch(self):
        """Thermal camera branch"""
        print("Building thermal branch...")
        
        # V4L2 source
        src = Gst.ElementFactory.make("v4l2src", "thermal-src")
        src.set_property("device", "/dev/video0")
        
        # Caps
        caps_filter = Gst.ElementFactory.make("capsfilter", "thermal-caps")
        caps = Gst.Caps.from_string("video/x-raw, format=UYVY, width=640, height=512, framerate=30/1")
        caps_filter.set_property("caps", caps)
        
        # Conversion
        convert = Gst.ElementFactory.make("videoconvert", "thermal-convert")
        queue = Gst.ElementFactory.make("queue", "thermal-queue")
        queue.set_property("max-size-buffers", 2)
        queue.set_property("leaky", 2)
        
        # Convert to NVMM
        nvconvert = Gst.ElementFactory.make("nvvideoconvert", "thermal-nvconvert")
        nvmm_filter = Gst.ElementFactory.make("capsfilter", "thermal-nvmm-caps")
        nvmm_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        nvmm_filter.set_property("caps", nvmm_caps)
        
        # Mux
        mux = Gst.ElementFactory.make("nvstreammux", "thermal-mux")
        mux.set_property("width", 640)
        mux.set_property("height", 512)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        mux.set_property("batched-push-timeout", 40000)
        
        # Inference
        infer = Gst.ElementFactory.make("nvinfer", "thermal-infer")
        infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
        infer.set_property("gpu-id", 0)
        
        # Queue
        infer_queue = Gst.ElementFactory.make("queue", "thermal-infer-queue")
        infer_queue.set_property("max-size-buffers", 2)
        infer_queue.set_property("leaky", 2)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "thermal-osd")
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.thermal_fps_probe, None)
        
        # Fakesink
        sink = Gst.ElementFactory.make("fakesink", "thermal-sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        
        # Add and link
        elements = [src, caps_filter, convert, queue, nvconvert, nvmm_filter, 
                   mux, infer, infer_queue, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)
        
        src.link(caps_filter)
        caps_filter.link(convert)
        convert.link(queue)
        queue.link(nvconvert)
        nvconvert.link(nvmm_filter)
        
        src_pad = nvmm_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        mux.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        return True
    
    def daylight_fps_probe(self, pad, info, user_data):
        self.daylight_frame_count += 1
        return Gst.PadProbeReturn.OK
    
    def thermal_fps_probe(self, pad, info, user_data):
        self.thermal_frame_count += 1
        return Gst.PadProbeReturn.OK
    
    def print_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.daylight_fps = self.daylight_frame_count / elapsed
            self.thermal_fps = self.thermal_frame_count / elapsed
            print(f"\rDaylight: {self.daylight_fps:.1f} FPS | Thermal: {self.thermal_fps:.1f} FPS", end='', flush=True)
            self.daylight_frame_count = 0
            self.thermal_frame_count = 0
            self.start_time = time.time()
        return True
    
    def bus_call(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\nEnd-of-stream")
            self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\nError: {err}: {debug}")
            self.loop.quit()
        return True
    
    def run(self):
        # Build branches
        if not self.build_daylight_branch():
            print("Failed to build daylight branch")
            return False
        
        if not self.build_thermal_branch():
            print("Failed to build thermal branch")
            return False
        
        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call)
        
        # FPS timer
        GLib.timeout_add(1000, self.print_fps)
        
        # Start
        print("Starting pipeline (NO TILING)...")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        self.pipeline.set_state(Gst.State.NULL)
        return True

if __name__ == "__main__":
    app = DualCamNoTiling()
    sys.exit(0 if app.run() else 1)
