#!/usr/bin/env python3
"""
Daylight Camera with Standard YOLO11 Inference (NO PREPROCESSING/TILING)
Uses the b8 model but on full 1920x1080 frame (no tiling)
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DaylightStandardInference:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("daylight-standard")
        self.loop = GLib.MainLoop()
        
        self.frame_count = 0
        self.start_time = None
        self.last_fps_print = time.time()
        
    def build_pipeline(self):
        """Build daylight camera with standard inference (no tiling)"""
        # Source: CSI camera
        src = Gst.ElementFactory.make("nvarguscamerasrc", "src")
        src.set_property("sensor-id", 0)
        
        # Caps: 1920x1080@30
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), width=1920, height=1080, "
            "format=NV12, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps")
        caps_filter.set_property("caps", caps)
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "mux")
        mux.set_property("width", 1920)
        mux.set_property("height", 1080)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        
        # Direct inference WITHOUT preprocessing/tiling
        # Uses full 1920x1080 frame
        infer = Gst.ElementFactory.make("nvinfer", "infer")
        infer.set_property("config-file-path", "config_infer_primary_fullframe.txt")
        
        # Queue after inference
        infer_queue = Gst.ElementFactory.make("queue", "infer-queue")
        infer_queue.set_property("max-size-buffers", 2)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "osd")
        
        # Add probe for FPS
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.fps_probe, None)
        
        # Sink (no display conflicts)
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        
        # Add elements
        for elem in [src, caps_filter, mux, infer, infer_queue, osd, sink]:
            self.pipeline.add(elem)
        
        # Link camera to mux
        src.link(caps_filter)
        src_pad = caps_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        # Link pipeline
        mux.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        print("✓ Pipeline built")
        return True
    
    def fps_probe(self, pad, info, user_data):
        """Measure FPS"""
        if self.start_time is None:
            self.start_time = time.time()
            print("✓ First frame received!")
        
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
            print("EOS")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"✗ Error: {err}")
            print(f"  {debug}")
            loop.quit()
        return True
    
    def run(self):
        """Run for 60 seconds"""
        print("=== Daylight Camera - Standard Inference (Full Frame) ===\n")
        
        if not self.build_pipeline():
            return False
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("✗ Failed to start")
            return False
        
        print("Running (60 second timeout)...\n")
        
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
            return True
        else:
            print("✗ No frames processed")
            return False

if __name__ == "__main__":
    test = DaylightStandardInference()
    success = test.run()
    sys.exit(0 if success else 1)
