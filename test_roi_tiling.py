#!/usr/bin/env python3
"""
Test ROI-based tiling with DeepStream native nvdspreprocess
This replaces the custom CUDA tiling kernel with native DeepStream ROI processing
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import time

# Initialize GStreamer
Gst.init(None)

class RoiTilingTest:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.fps_start_time = time.time()
        self.frame_count = 0
        
    def create_pipeline(self):
        """Create pipeline with ROI-based preprocessing"""
        print("\n=== Creating ROI-Based Tiling Pipeline ===")
        print("Config: config_preprocess_roi_native.txt")
        print("Model: model_b8_gpu0_fp16.engine (batch=8)")
        print("Expected: 8 ROIs extracted natively by DeepStream\n")
        
        # Create pipeline
        self.pipeline = Gst.Pipeline.new("roi-tiling-test")
        
        # Camera source (CSI daylight)
        source = Gst.ElementFactory.make("nvarguscamerasrc", "csi-camera")
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
        
        # Preprocess with ROI (native DeepStream)
        preprocess = Gst.ElementFactory.make("nvdspreprocess", "preprocess")
        preprocess.set_property("config-file", "config_preprocess_roi_native.txt")
        
        # Queue before inference
        preprocess_queue = Gst.ElementFactory.make("queue", "preprocess-queue")
        preprocess_queue.set_property("max-size-buffers", 2)
        preprocess_queue.set_property("leaky", 2)
        
        # Inference
        infer = Gst.ElementFactory.make("nvinfer", "infer")
        infer.set_property("config-file-path", "config_infer_primary_yolo11_tiling.txt")
        infer.set_property("input-tensor-meta", True)
        infer.set_property("gpu-id", 0)
        
        # Queue after inference
        infer_queue = Gst.ElementFactory.make("queue", "infer-queue")
        infer_queue.set_property("max-size-buffers", 2)
        infer_queue.set_property("leaky", 2)
        
        # OSD for visualization
        osd = Gst.ElementFactory.make("nvdsosd", "osd")
        osd.set_property("gpu-id", 0)
        osd.set_property("process-mode", 0)
        
        # Fakesink (no display)
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("async", False)
        
        # Add elements
        elements = [source, caps_src, streammux, preprocess, preprocess_queue,
                   infer, infer_queue, osd, sink]
        for element in elements:
            if not element:
                print(f"ERROR: Failed to create element!")
                return False
            self.pipeline.add(element)
        
        # Link source to streammux
        source.link(caps_src)
        
        sinkpad = streammux.get_request_pad("sink_0")
        srcpad = caps_src.get_static_pad("src")
        srcpad.link(sinkpad)
        
        # Link rest of pipeline
        if not streammux.link(preprocess):
            print("ERROR: Failed to link streammux → preprocess")
            return False
            
        if not preprocess.link(preprocess_queue):
            print("ERROR: Failed to link preprocess → queue")
            return False
            
        if not preprocess_queue.link(infer):
            print("ERROR: Failed to link queue → infer")
            return False
            
        if not infer.link(infer_queue):
            print("ERROR: Failed to link infer → queue")
            return False
            
        if not infer_queue.link(osd):
            print("ERROR: Failed to link queue → osd")
            return False
            
        if not osd.link(sink):
            print("ERROR: Failed to link osd → sink")
            return False
        
        # Add probe to OSD for frame counting
        osd_src_pad = osd.get_static_pad("src")
        osd_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.osd_probe, None)
        
        # Bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)
        
        print("✓ Pipeline created successfully")
        return True
    
    def osd_probe(self, pad, info, user_data):
        """Count frames and print FPS"""
        self.frame_count += 1
        
        # Print FPS every 30 frames
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            fps = 30.0 / elapsed
            print(f"Frame {self.frame_count:4d} | FPS: {fps:5.1f} | Elapsed: {int(elapsed * self.frame_count / 30):3d}s")
            self.fps_start_time = time.time()
        
        return Gst.PadProbeReturn.OK
    
    def on_bus_message(self, bus, message):
        """Handle bus messages"""
        t = message.type
        
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\n❌ ERROR: {err}")
            print(f"Debug: {debug}")
            self.loop.quit()
            
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"\n⚠️  WARNING: {warn}")
            
        elif t == Gst.MessageType.EOS:
            print("\n✓ End of stream")
            self.loop.quit()
            
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old, new, pending = message.parse_state_changed()
                print(f"Pipeline state: {old.value_nick} → {new.value_nick}")
    
    def run(self):
        """Run the pipeline"""
        if not self.create_pipeline():
            print("Failed to create pipeline")
            return False
        
        print("\n=== Starting Pipeline ===")
        print("Testing ROI-based tiling (native DeepStream)")
        print("Target: 10 minutes stable operation")
        print("Press Ctrl+C to stop\n")
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Unable to set pipeline to PLAYING state")
            return False
        
        # Run main loop
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\n=== Interrupted by user ===")
        
        # Cleanup
        print("\nStopping pipeline...")
        self.pipeline.set_state(Gst.State.NULL)
        
        total_time = time.time() - self.fps_start_time
        print(f"\n=== Test Summary ===")
        print(f"Total frames: {self.frame_count}")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Average FPS: {self.frame_count / total_time:.1f}")
        
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("ROI-Based Tiling Test (Native DeepStream)")
    print("Alternative to custom CUDA kernel")
    print("=" * 60)
    
    test = RoiTilingTest()
    success = test.run()
    
    sys.exit(0 if success else 1)
