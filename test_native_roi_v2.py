#!/usr/bin/env python3
"""
Test native DeepStream ROI tiling (no CUDA kernel)
Uses config_preprocess_roi_native_v2.txt with process-mode=0
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import time

Gst.init(None)

class NativeRoiPipeline:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.start_time = None
        
    def create_pipeline(self):
        """Create pipeline with native ROI tiling"""
        print("\n=== Native ROI Tiling Test ===")
        print("Config: config_preprocess_roi_native_v2.txt")
        print("Using process-mode=0 with roi-params-src-0")
        print("NO CUDA kernel - pure DeepStream native!\n")
        
        self.pipeline = Gst.Pipeline.new("native-roi-test")
        
        # Camera source
        source = Gst.ElementFactory.make("nvarguscamerasrc", "camera")
        source.set_property("sensor-id", 0)
        
        caps_src = Gst.ElementFactory.make("capsfilter", "src-caps")
        caps_src.set_property("caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12"))
        
        # Streammux
        streammux = Gst.ElementFactory.make("nvstreammux", "mux")
        streammux.set_property("width", 1920)
        streammux.set_property("height", 1080)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 33333)
        streammux.set_property("live-source", 1)
        streammux.set_property("gpu-id", 0)
        
        # Native ROI preprocessing
        preprocess = Gst.ElementFactory.make("nvdspreprocess", "preprocess")
        preprocess.set_property("config-file", "config_preprocess_roi_native_v2.txt")
        
        # Queue
        queue = Gst.ElementFactory.make("queue", "queue")
        queue.set_property("max-size-buffers", 2)
        queue.set_property("leaky", 2)
        
        # Inference
        infer = Gst.ElementFactory.make("nvinfer", "infer")
        infer.set_property("config-file-path", "config_infer_primary_yolo11_tiling.txt")
        infer.set_property("input-tensor-meta", True)
        infer.set_property("gpu-id", 0)
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "osd")
        osd.set_property("gpu-id", 0)
        
        # Sink
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        
        # Add elements
        for elem in [source, caps_src, streammux, preprocess, queue, infer, osd, sink]:
            if not elem:
                print(f"ERROR: Failed to create element")
                return False
            self.pipeline.add(elem)
        
        # Link source to mux
        source.link(caps_src)
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = caps_src.get_static_pad("src")
        srcpad.link(sinkpad)
        
        # Link rest
        if not streammux.link(preprocess):
            print("ERROR: mux → preprocess")
            return False
        if not preprocess.link(queue):
            print("ERROR: preprocess → queue")
            return False
        if not queue.link(infer):
            print("ERROR: queue → infer")
            return False
        if not infer.link(osd):
            print("ERROR: infer → osd")
            return False
        if not osd.link(sink):
            print("ERROR: osd → sink")
            return False
        
        # Probe
        osd_pad = osd.get_static_pad("src")
        osd_pad.add_probe(Gst.PadProbeType.BUFFER, self.probe, None)
        
        # Bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)
        
        print("✓ Pipeline created\n")
        return True
    
    def probe(self, pad, info, user_data):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            fps = 30.0 / elapsed
            total = time.time() - self.start_time
            print(f"Frame {self.frame_count:4d} | FPS: {fps:5.1f} | Time: {int(total):3d}s")
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
        
        print("=== Starting Native ROI Tiling ===")
        print("Target: 10 minutes stable")
        print("If this works, no CUDA debugging needed!\n")
        
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
        print(f"\n=== Results ===")
        print(f"Frames: {self.frame_count}")
        print(f"Time: {total:.1f}s ({total/60:.1f} min)")
        print(f"Avg FPS: {self.frame_count/total:.1f}")
        
        if total >= 180:
            print("\n✅ SUCCESS: Ran 3+ minutes - native ROI works!")
        
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("Native DeepStream ROI Tiling Test")
    print("No CUDA Kernel - Pure DeepStream")
    print("=" * 60)
    
    pipeline = NativeRoiPipeline()
    success = pipeline.run()
    
    sys.exit(0 if success else 1)
