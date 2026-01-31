#!/usr/bin/env python3
"""
Simplified tiling using GStreamer nvvideoconvert crop + tee
Leverages NVIDIA VIC engine for GPU-accelerated cropping/scaling
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import time

Gst.init(None)

class NvTilingPipeline:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.start_time = None
        
    def create_pipeline(self):
        """
        Create pipeline using nvvideoconvert cropping (GPU-accelerated via VIC)
        
        Simplified approach: Test with fewer tiles first (2 tiles)
        """
        print("\n=== Creating NV Tiling Pipeline ===")
        print("Using nvvideoconvert crop (GPU VIC engine)")
        print("Test: 2 tiles first, then scale to 8\n")
        
        self.pipeline = Gst.Pipeline.new("nv-tiling")
        
        # Source
        source = Gst.ElementFactory.make("nvarguscamerasrc", "camera")
        source.set_property("sensor-id", 0)
        
        caps_src = Gst.ElementFactory.make("capsfilter", "src-caps")
        caps_src.set_property("caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12"))
        
        # Tee
        tee = Gst.ElementFactory.make("tee", "tee")
        
        # Streammux (batch=2 for testing)
        streammux = Gst.ElementFactory.make("nvstreammux", "mux")
        streammux.set_property("width", 640)
        streammux.set_property("height", 640)
        streammux.set_property("batch-size", 2)  # Start with 2 tiles
        streammux.set_property("batched-push-timeout", 33333)
        streammux.set_property("live-source", 1)
        streammux.set_property("gpu-id", 0)
        
        # Fakesink (no inference for now - just test tiling)
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        
        # Add elements
        for elem in [source, caps_src, tee, streammux, sink]:
            if not elem:
                print("ERROR: Failed to create element")
                return False
            self.pipeline.add(elem)
        
        # Link source → tee
        if not source.link(caps_src) or not caps_src.link(tee):
            print("ERROR: Failed to link source → tee")
            return False
        
        # Create 2 tile branches for testing
        tiles = [(0, 0), (640, 0)]  # Top-left and top-middle
        
        for i, (x, y) in enumerate(tiles):
            queue = Gst.ElementFactory.make("queue", f"q-{i}")
            queue.set_property("max-size-buffers", 2)
            
            # nvvideoconvert with cropping
            nvconv = Gst.ElementFactory.make("nvvideoconvert", f"conv-{i}")
            nvconv.set_property("left", x)
            nvconv.set_property("top", y)
            nvconv.set_property("right", 1920 - x - 640)
            nvconv.set_property("bottom", 1080 - y - 640)
            
            caps = Gst.ElementFactory.make("capsfilter", f"caps-{i}")
            caps.set_property("caps",
                Gst.Caps.from_string("video/x-raw(memory:NVMM), width=640, height=640, format=NV12"))
            
            for elem in [queue, nvconv, caps]:
                if not elem:
                    print(f"ERROR: Failed to create element for tile {i}")
                    return False
                self.pipeline.add(elem)
            
            # Link tee → branch
            tee_src = tee.get_request_pad(f"src_{i}")
            queue_sink = queue.get_static_pad("sink")
            tee_src.link(queue_sink)
            
            # Link branch
            if not queue.link(nvconv) or not nvconv.link(caps):
                print(f"ERROR: Failed to link branch {i}")
                return False
            
            # Link to mux
            mux_sink = streammux.get_request_pad(f"sink_{i}")
            caps_src_pad = caps.get_static_pad("src")
            caps_src_pad.link(mux_sink)
            
            print(f"  ✓ Tile {i}: crop({x},{y},640,640)")
        
        # Link mux → sink
        if not streammux.link(sink):
            print("ERROR: Failed to link mux → sink")
            return False
        
        # Add probe
        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.probe, None)
        
        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)
        
        print("\n✓ Pipeline created")
        return True
    
    def probe(self, pad, info, user_data):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            fps = 30.0 / elapsed
            total = time.time() - self.start_time
            print(f"Frame {self.frame_count:4d} | FPS: {fps:5.1f} | Time: {int(total)}s")
            self.fps_start_time = time.time()
        return Gst.PadProbeReturn.OK
    
    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n❌ ERROR: {err}\n{debug}")
            self.loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, _ = message.parse_warning()
            print(f"⚠️  {warn}")
        elif t == Gst.MessageType.EOS:
            self.loop.quit()
    
    def run(self):
        if not self.create_pipeline():
            return False
        
        print("\n=== Starting ===\n")
        
        self.start_time = time.time()
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Can't start")
            return False
        
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\n=== Stopped ===")
        
        self.pipeline.set_state(Gst.State.NULL)
        
        total = time.time() - self.start_time
        print(f"\nFrames: {self.frame_count} | Time: {total:.1f}s | FPS: {self.frame_count/total:.1f}")
        
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("NV Tiling Test (2 tiles)")
    print("=" * 60)
    
    pipeline = NvTilingPipeline()
    pipeline.run()
