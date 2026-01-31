#!/usr/bin/env python3
"""
Daylight Camera Test WITH DISPLAY
Simple camera stream with on-screen display
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DaylightDisplayTest:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("daylight-display")
        self.loop = GLib.MainLoop()
        
        self.frame_count = 0
        self.start_time = None
        self.last_fps_print = time.time()
        
    def build_pipeline(self):
        """Build camera with display"""
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
        
        # Convert to CPU memory for display
        nvconv = Gst.ElementFactory.make("nvvideoconvert", "nvconv")
        
        # Convert format
        conv = Gst.ElementFactory.make("videoconvert", "conv")
        
        # Add probe for FPS
        probe_pad = conv.get_static_pad("src")
        probe_pad.add_probe(Gst.PadProbeType.BUFFER, self.fps_probe, None)
        
        # Display sink
        sink = Gst.ElementFactory.make("nveglglessink", "sink")
        sink.set_property("sync", False)
        
        # Add elements
        for elem in [src, caps_filter, nvconv, conv, sink]:
            self.pipeline.add(elem)
        
        # Link
        src.link(caps_filter)
        caps_filter.link(nvconv)
        nvconv.link(conv)
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
        print("=== Daylight Camera Display Test ===\n")
        
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
    test = DaylightDisplayTest()
    success = test.run()
    sys.exit(0 if success else 1)
