#!/usr/bin/env python3
"""
Simple Camera Test - No Inference
Tests if camera is accessible and streaming
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class SimpleCameraTest:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("simple-camera")
        self.loop = GLib.MainLoop()
        self.frame_count = 0
        self.start_time = None
        
    def build_pipeline(self):
        """Build simple camera → fakesink pipeline"""
        # Source: CSI camera
        src = Gst.ElementFactory.make("nvarguscamerasrc", "src")
        if not src:
            print("ERROR: Failed to create nvarguscamerasrc - ARGUS driver may not be available")
            return False
        src.set_property("sensor-id", 0)
        
        # Caps: 1920x1080@30
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), width=1920, height=1080, "
            "format=NV12, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "caps")
        caps_filter.set_property("caps", caps)
        
        # Simple sink
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", True)
        
        # Add probe to count frames
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.frame_probe, None)
        
        # Add elements
        for elem in [src, caps_filter, sink]:
            self.pipeline.add(elem)
        
        # Link
        src.link(caps_filter)
        caps_filter.link(sink)
        
        print("✓ Pipeline built successfully")
        return True
    
    def frame_probe(self, pad, info, user_data):
        """Count frames"""
        if self.start_time is None:
            self.start_time = time.time()
            print("✓ First frame received!")
        
        self.frame_count += 1
        
        # Print FPS every 1 second
        elapsed = time.time() - self.start_time
        if self.frame_count % 30 == 0:
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"  Frames: {self.frame_count} | FPS: {fps:.1f}")
        
        return Gst.PadProbeReturn.OK
    
    def bus_call(self, bus, message, loop):
        """Handle messages"""
        t = message.type
        if t == Gst.MessageType.STATE_CHANGED:
            old, new, pending = message.parse_state_changed()
            print(f"  State: {old.value_nick} → {new.value_nick}")
        elif t == Gst.MessageType.EOS:
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
        """Run test for 30 seconds"""
        print("=== Simple Camera Test ===\n")
        
        if not self.build_pipeline():
            return False
        
        # Setup bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        # Start pipeline
        print("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("✗ Failed to start pipeline")
            return False
        
        print("Waiting for frames (30 second timeout)...\n")
        
        # Run with timeout
        try:
            GLib.timeout_add_seconds(30, lambda: self.loop.quit())
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        # Cleanup
        self.pipeline.set_state(Gst.State.NULL)
        
        print(f"\n=== Results ===")
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"✓ Success! Received {self.frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
            return True
        else:
            print("✗ No frames received - camera not accessible")
            return False

if __name__ == "__main__":
    test = SimpleCameraTest()
    success = test.run()
    sys.exit(0 if success else 1)
