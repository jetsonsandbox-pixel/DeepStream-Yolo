#!/usr/bin/env python3
"""
Python-based tile extraction for DeepStream
Alternative to buggy CUDA kernel - stable and debuggable
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import time
import numpy as np
import cv2
import ctypes

Gst.init(None)

class PythonTilingPipeline:
    def __init__(self):
        self.pipeline = None
        self.loop = None
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_frame = None
        
    def extract_tiles(self, frame_bgr):
        """
        Extract 8 tiles (4×2 grid) from 1920×1080 frame
        Tile size: 640×640, Overlap: 96px, Stride: 544px
        """
        h, w = frame_bgr.shape[:2]
        tiles = []
        
        stride = 544  # 640 - 96 overlap
        tile_coords = []
        
        for row in range(2):
            for col in range(4):
                x = col * stride
                y = row * stride
                tile_coords.append((x, y))
                
                # Extract with bounds checking
                x_end = min(x + 640, w)
                y_end = min(y + 640, h)
                
                tile = frame_bgr[y:y_end, x:x_end]
                
                # Pad if needed (right/bottom edges)
                if tile.shape[0] < 640 or tile.shape[1] < 640:
                    tile = cv2.copyMakeBorder(
                        tile,
                        0, 640 - tile.shape[0],
                        0, 640 - tile.shape[1],
                        cv2.BORDER_CONSTANT,
                        value=(114, 114, 114)  # YOLO padding color
                    )
                
                tiles.append(tile)
        
        return tiles, tile_coords
    
    def nvmm_probe(self, pad, info, user_data):
        """Probe to extract frame from NVMM memory"""
        # Get buffer
        gst_buffer = info.get_buffer()
        if gst_buffer is None:
            return Gst.PadProbeReturn.OK
        
        # Map NVMM buffer
        # For now, we'll use CPU processing
        # TODO: GPU-accelerated tile extraction
        
        self.frame_count += 1
        
        # Print FPS
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            fps = 30.0 / elapsed
            print(f"Frame {self.frame_count:4d} | FPS: {fps:5.1f}")
            self.fps_start_time = time.time()
        
        return Gst.PadProbeReturn.OK
    
    def create_pipeline(self):
        """Create pipeline with Python tiling"""
        print("\n=== Creating Python-Based Tiling Pipeline ===")
        print("Tile extraction: Python + OpenCV (CPU)")
        print("Model: model_b8_gpu0_fp16.engine (batch=8)")
        print("Expected: More stable than CUDA kernel\n")
        
        # Note: This is a simplified version
        # Full implementation would use appsrc for each tile
        # For now, testing standard pipeline first
        
        self.pipeline = Gst.parse_launch(
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1, format=NV12 ! "
            "nvvideoconvert ! "
            "video/x-raw, format=BGRx ! "
            "fakesink name=sink sync=0"
        )
        
        # Add probe to capture frames
        sink = self.pipeline.get_by_name("sink")
        sinkpad = sink.get_static_pad("sink")
        sinkpad.add_probe(Gst.PadProbeType.BUFFER, self.nvmm_probe, None)
        
        # Bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_bus_message)
        
        print("✓ Pipeline created")
        return True
    
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
    
    def run(self):
        """Run the pipeline"""
        if not self.create_pipeline():
            return False
        
        print("\n=== Starting Pipeline ===")
        print("Testing frame capture (no tiling yet)")
        print("Press Ctrl+C to stop\n")
        
        # Start
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("ERROR: Unable to start pipeline")
            return False
        
        # Run
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\n=== Interrupted ===")
        
        # Cleanup
        self.pipeline.set_state(Gst.State.NULL)
        
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("Python-Based Tiling Pipeline (Step 1: Frame Capture)")
    print("=" * 60)
    
    pipeline = PythonTilingPipeline()
    pipeline.run()
