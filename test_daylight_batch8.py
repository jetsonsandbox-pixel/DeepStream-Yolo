#!/usr/bin/env python3
"""
Test: Daylight Camera Only with Batch=8 Tiling
Tests if daylight branch alone can handle batch=8 without thermal camera
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DaylightOnlyTest:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("daylight-only")
        self.loop = GLib.MainLoop()
        
        self.frame_count = 0
        self.start_time = None
        self.last_fps_print = time.time()
        
    def build_pipeline(self):
        """Build daylight CSI branch with tiling (batch=8)"""
        print("Building daylight branch with batch=8 tiling...")
        
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
        
        # Preprocessing with tiling
        preprocess = Gst.ElementFactory.make("nvdspreprocess", "daylight-preprocess")
        preprocess.set_property("config-file", "config_preprocess_tiling.txt")
        
        # Queue after preprocessing
        preprocess_queue = Gst.ElementFactory.make("queue", "daylight-preprocess-queue")
        preprocess_queue.set_property("max-size-buffers", 8)
        preprocess_queue.set_property("max-size-bytes", 0)
        preprocess_queue.set_property("max-size-time", 0)
        
        # Inference with daylight model (batch=8)
        infer = Gst.ElementFactory.make("nvinfer", "daylight-infer")
        infer.set_property("config-file-path", "config_infer_primary_yolo11_tiling.txt")
        
        # Queue after inference
        infer_queue = Gst.ElementFactory.make("queue", "daylight-infer-queue")
        infer_queue.set_property("max-size-buffers", 4)
        infer_queue.set_property("max-size-bytes", 0)
        infer_queue.set_property("max-size-time", 0)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "daylight-osd")
        
        # Add probe for FPS
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.fps_probe, None)
        
        # Sink - fakesink to avoid display conflicts
        sink = Gst.ElementFactory.make("fakesink", "daylight-sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        
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
        mux.link(preprocess)
        preprocess.link(preprocess_queue)
        preprocess_queue.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        print("✓ Pipeline built successfully")
        return True
    
    def fps_probe(self, pad, info, user_data):
        """Measure FPS"""
        if self.start_time is None:
            self.start_time = time.time()
            print("✓ First frame processed!")
        
        self.frame_count += 1
        
        current_time = time.time()
        if current_time - self.last_fps_print >= 5.0:
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"  Daylight FPS: {fps:.1f} (frames: {self.frame_count})")
            self.last_fps_print = current_time
        
        return Gst.PadProbeReturn.OK
    
    def bus_call(self, bus, message, loop):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"✗ Error: {err}")
            print(f"  Debug: {debug}")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"⚠ Warning: {err}")
        return True
    
    def run(self):
        """Run test for 120 seconds"""
        print("=== Daylight Camera Test: Batch=8 Tiling ===\n")
        
        if not self.build_pipeline():
            return False
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("✗ Failed to start pipeline")
            return False
        
        print("Running (120 second timeout)...\n")
        
        try:
            GLib.timeout_add_seconds(120, lambda: self.loop.quit())
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        self.pipeline.set_state(Gst.State.NULL)
        
        print(f"\n=== Results ===")
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"✓ Success! Processed {self.frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
            return True
        else:
            print("✗ No frames processed")
            return False

if __name__ == "__main__":
    test = DaylightOnlyTest()
    success = test.run()
    sys.exit(0 if success else 1)
