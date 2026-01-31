#!/usr/bin/env python3
"""
Thermal Camera Test WITH DISPLAY
USB thermal camera stream with on-screen display
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class ThermalDisplayTest:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("thermal-display")
        self.loop = GLib.MainLoop()
        
        self.frame_count = 0
        self.start_time = None
        self.last_fps_print = time.time()
        
    def build_pipeline(self):
        """Build USB thermal camera with display"""
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
        conv = Gst.ElementFactory.make("videoconvert", "conv")
        
        # Add probe for FPS
        probe_pad = conv.get_static_pad("src")
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, self.fps_probe, None)
        
        # Display sink
        sink = Gst.ElementFactory.make("nveglglessink", "sink")
        sink.set_property("sync", False)
        
        # Add elements
        for elem in [src, caps_filter, conv, sink]:
            self.pipeline.add(elem)
        
        # Link
        src.link(caps_filter)
        caps_filter.link(conv)
        conv.link(sink)
        
        print("✓ Display pipeline built")
        return True
    
    def fps_probe(self, pad, info, user_data):
        """Measure FPS"""
        if self.start_time is None:
            self.start_time = time.time()
            print("✓ Display window opened!")
        
        self.frame_count += 1
        
        current_time = time.time()
        if current_time - self.last_fps_print >= 5.0:
            elapsed = current_time - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"  FPS: {fps:.1f}")
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
            loop.quit()
        return True
    
    def run(self):
        """Run for 60 seconds"""
        print("=== Thermal Camera Display Test ===\n")
        
        if not self.build_pipeline():
            return False
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print("Starting pipeline...")
        print("(Display window should appear in 5-10 seconds)\n")
        
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
        
        print(f"\n✓ Test complete")
        return True

if __name__ == "__main__":
    test = ThermalDisplayTest()
    success = test.run()
    sys.exit(0 if success else 1)
