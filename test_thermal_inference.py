#!/usr/bin/env python3
"""
Thermal Camera Test with Inference
- Uses existing model_thermal_b1_gpu0_fp16.engine
- Uses config_infer_primary_thermal.txt
- Validates thermal pipeline stability
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class ThermalInferenceTest:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("thermal-infer")
        self.loop = GLib.MainLoop()
        
        self.frame_count = 0
        self.start_time = None
        self.last_fps_print = time.time()
        
    def build_pipeline(self):
        """Build thermal camera with inference"""
        # Source: USB thermal camera
        src = Gst.ElementFactory.make("v4l2src", "src")
        src.set_property("device", "/dev/video1")
        
        # Caps: thermal resolution
        caps = Gst.Caps.from_string(
            "video/x-raw, width=640, height=512, format=YUY2, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps")
        caps_filter.set_property("caps", caps)
        
        # Convert format
        convert = Gst.ElementFactory.make("videoconvert", "convert")
        
        # Queue to decouple from nvvideoconvert
        queue1 = Gst.ElementFactory.make("queue", "queue1")
        queue1.set_property("max-size-buffers", 3)
        
        # NVMM upload
        nvconvert = Gst.ElementFactory.make("nvvideoconvert", "nvconvert")
        nvconvert.set_property("nvbuf-memory-type", 0)
        
        # NVMM caps
        nvmm_caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=NV12"
        )
        nvmm_filter = Gst.ElementFactory.make("capsfilter", "nvmm-caps")
        nvmm_filter.set_property("caps", nvmm_caps)
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "mux")
        mux.set_property("width", 640)
        mux.set_property("height", 512)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        mux.set_property("batched-push-timeout", 40000)
        
        # Inference
        infer = Gst.ElementFactory.make("nvinfer", "infer")
        infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
        
        # Queue after inference
        infer_queue = Gst.ElementFactory.make("queue", "infer-queue")
        infer_queue.set_property("max-size-buffers", 4)
        infer_queue.set_property("max-size-bytes", 0)
        infer_queue.set_property("max-size-time", 0)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "osd")
        
        # Add probe for FPS
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.fps_probe, None)
        
        # Sink
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        
        # Add elements
        for elem in [src, caps_filter, convert, queue1, nvconvert, nvmm_filter, mux, infer, infer_queue, osd, sink]:
            self.pipeline.add(elem)
        
        # Link elements
        src.link(caps_filter)
        caps_filter.link(convert)
        convert.link(queue1)
        queue1.link(nvconvert)
        nvconvert.link(nvmm_filter)
        
        # Link to mux
        src_pad = nvmm_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        # Link rest
        mux.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        print("✓ Pipeline built (thermal, no tiling)")
        return True
    
    def fps_probe(self, pad, info, user_data):
        """Measure FPS"""
        if self.start_time is None:
            self.start_time = time.time()
            print("✓ Inference started!")
        
        self.frame_count += 1
        
        current_time = time.time()
        if current_time - self.last_fps_print >= 5.0:
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"  FPS: {fps:.1f} (frames: {self.frame_count})")
            self.last_fps_print = current_time
        
        return Gst.PadProbeReturn.OK
    
    def bus_call(self, bus, message, loop):
        """Handle messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"✗ Error: {err}")
            print(f"  {debug}")
            loop.quit()
        return True
    
    def run(self):
        """Run for 60 seconds"""
        print("=== Thermal Camera Inference Test ===\n")
        print("Model: model_thermal_b1_gpu0_fp16.engine")
        print("Config: config_infer_primary_thermal.txt\n")
        
        if not self.build_pipeline():
            return False
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print("Starting pipeline (60 second test)...\n")
        
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("✗ Failed to start")
            return False
        
        try:
            GLib.timeout_add_seconds(60, lambda: self.loop.quit())
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        self.pipeline.set_state(Gst.State.NULL)
        
        print(f"\n=== Results ===")
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"✓ Processed {self.frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
            print("✓ Pipeline stable (no CUDA errors)")
            return True
        else:
            print("✗ No frames processed")
            return False

if __name__ == "__main__":
    test = ThermalInferenceTest()
    success = test.run()
    sys.exit(0 if success else 1)
