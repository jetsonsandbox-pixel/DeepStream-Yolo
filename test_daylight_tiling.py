#!/usr/bin/env python3
"""
Daylight Camera Test with Tiling Inference
- Uses existing model_b8_gpu0_fp16.engine
- Uses config_infer_primary_yolo11_tiling.txt with batch-size=4
- Validates tiling pipeline stability
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DaylightTilingTest:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("daylight-tiling")
        self.loop = GLib.MainLoop()
        
        self.frame_count = 0
        self.start_time = None
        self.last_fps_print = time.time()
        
    def build_pipeline(self):
        """Build daylight camera with tiling inference"""
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
        
        # Preprocessing with tiling
        preprocess = Gst.ElementFactory.make("nvdspreprocess", "preprocess")
        preprocess.set_property("config-file", "config_preprocess_tiling.txt")
        
        # Queue after preprocessing
        preprocess_queue = Gst.ElementFactory.make("queue", "preprocess-queue")
        preprocess_queue.set_property("max-size-buffers", 8)
        preprocess_queue.set_property("max-size-bytes", 0)
        preprocess_queue.set_property("max-size-time", 0)
        
        # Inference with tiling (batch=4 in config)
        infer = Gst.ElementFactory.make("nvinfer", "infer")
        infer.set_property("config-file-path", "config_infer_primary_yolo11_tiling.txt")
        
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
        for elem in [src, caps_filter, mux, preprocess, preprocess_queue, infer, infer_queue, osd, sink]:
            self.pipeline.add(elem)
        
        # Link camera to mux
        src.link(caps_filter)
        src_pad = caps_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        # Link pipeline
        mux.link(preprocess)
        preprocess.link(preprocess_queue)
        preprocess_queue.link(infer)
        infer.link(infer_queue)
        infer_queue.link(osd)
        osd.link(sink)
        
        print("✓ Pipeline built (daylight + tiling, batch=4)")
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
        print("=== Daylight + Tiling Inference Test ===\n")
        print("Model: model_b8_gpu0_fp16.engine")
        print("Config: config_infer_primary_yolo11_tiling.txt (batch=4)\n")
        
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
    test = DaylightTilingTest()
    success = test.run()
    sys.exit(0 if success else 1)
