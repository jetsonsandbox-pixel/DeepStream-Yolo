#!/usr/bin/env python3
"""
Test Thermal Camera Branch Only
Single branch pipeline for thermal USB camera testing
"""

import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class ThermalCameraTest:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("thermal-test-pipeline")
        self.loop = GLib.MainLoop()
        
    def build_thermal_branch(self):
        """Build thermal USB branch without tiling"""
        print("Creating thermal camera pipeline elements...")
        
        # Source: USB camera
        src = Gst.ElementFactory.make("v4l2src", "thermal-src")
        if not src:
            print("ERROR: Failed to create v4l2src")
            return False
        src.set_property("device", "/dev/video1")
        
        # Caps: 640x512 YUYV
        caps = Gst.Caps.from_string(
            "video/x-raw, width=640, height=512, format=YUY2, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "thermal-caps")
        if not caps_filter:
            print("ERROR: Failed to create capsfilter")
            return False
        caps_filter.set_property("caps", caps)
        
        # Video convert
        convert = Gst.ElementFactory.make("videoconvert", "thermal-convert")
        if not convert:
            print("ERROR: Failed to create videoconvert")
            return False
        
        # NVMM upload
        nvconvert = Gst.ElementFactory.make("nvvideoconvert", "thermal-nvconvert")
        if not nvconvert:
            print("ERROR: Failed to create nvvideoconvert")
            return False
        
        # NVMM caps
        nvmm_caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), format=RGBA"
        )
        nvmm_filter = Gst.ElementFactory.make("capsfilter", "thermal-nvmm-caps")
        if not nvmm_filter:
            print("ERROR: Failed to create nvmm capsfilter")
            return False
        nvmm_filter.set_property("caps", nvmm_caps)
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "thermal-mux")
        if not mux:
            print("ERROR: Failed to create nvstreammux")
            return False
        mux.set_property("width", 640)
        mux.set_property("height", 512)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        
        # Inference with thermal model (no preprocessing)
        infer = Gst.ElementFactory.make("nvinfer", "thermal-infer")
        if not infer:
            print("ERROR: Failed to create nvinfer")
            return False
        infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "thermal-osd")
        if not osd:
            print("ERROR: Failed to create nvdsosd")
            return False
        
        # Sink
        sink = Gst.ElementFactory.make("nveglglessink", "thermal-sink")
        if not sink:
            print("ERROR: Failed to create nveglglessink")
            return False
        sink.set_property("sync", False)
        
        # Add elements to pipeline
        elements = [src, caps_filter, convert, nvconvert, nvmm_filter, 
                   mux, infer, osd, sink]
        print(f"Adding {len(elements)} elements to pipeline...")
        for elem in elements:
            self.pipeline.add(elem)
        
        # Link elements
        print("Linking pipeline elements...")
        if not src.link(caps_filter):
            print("ERROR: Failed to link src -> caps_filter")
            return False
        
        if not caps_filter.link(convert):
            print("ERROR: Failed to link caps_filter -> convert")
            return False
            
        if not convert.link(nvconvert):
            print("ERROR: Failed to link convert -> nvconvert")
            return False
            
        if not nvconvert.link(nvmm_filter):
            print("ERROR: Failed to link nvconvert -> nvmm_filter")
            return False
        
        # Link to mux sink pad
        src_pad = nvmm_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
            print("ERROR: Failed to link nvmm_filter -> mux")
            return False
        
        # Link rest of pipeline
        if not mux.link(infer):
            print("ERROR: Failed to link mux -> infer")
            return False
            
        if not infer.link(osd):
            print("ERROR: Failed to link infer -> osd")
            return False
            
        if not osd.link(sink):
            print("ERROR: Failed to link osd -> sink")
            return False
        
        print("✓ Thermal branch built successfully")
        return True
    
    def bus_call(self, bus, message, loop):
        """Handle bus messages"""
        t = message.type
        if t == Gst.MessageType.EOS:
            print("\nEnd-of-stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"\nERROR: {err}")
            print(f"Debug: {debug}")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"\nWARNING: {warn}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending = message.parse_state_changed()
                print(f"Pipeline state: {old_state.value_nick} -> {new_state.value_nick}")
        return True
    
    def run(self):
        """Build and run thermal test pipeline"""
        print("=" * 60)
        print("THERMAL CAMERA TEST")
        print("=" * 60)
        
        print("\nBuilding thermal branch...")
        if not self.build_thermal_branch():
            print("✗ Failed to build thermal branch")
            return False
        
        # Setup bus watch
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        # Start playing
        print("\nStarting pipeline...")
        print("Press Ctrl+C to stop\n")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("✗ Failed to set pipeline to PLAYING state")
            return False
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        # Cleanup
        print("Stopping pipeline...")
        self.pipeline.set_state(Gst.State.NULL)
        print("✓ Pipeline stopped")
        return True

if __name__ == "__main__":
    print("\nThermal Camera Test Pipeline")
    print("Device: /dev/video1")
    print("Resolution: 640x512")
    print("Model: config_infer_primary_thermal.txt\n")
    
    pipeline = ThermalCameraTest()
    sys.exit(0 if pipeline.run() else 1)
