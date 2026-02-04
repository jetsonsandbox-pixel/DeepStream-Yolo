#!/usr/bin/env python3
"""
Dual Camera DeepStream Pipeline - Stable Version
- Daylight CSI: 1920x1080, standard preprocessing (no custom tiling)
- Thermal USB: 640x512
Both use standard nvinfer without custom preprocessing
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class DualCameraPipeline:
    def __init__(self):
        Gst.init(None)
        self.pipeline = Gst.Pipeline.new("dual-camera-pipeline")
        self.loop = GLib.MainLoop()
        
        # FPS tracking
        self.daylight_frame_count = 0
        self.thermal_frame_count = 0
        self.daylight_start_time = None
        self.thermal_start_time = None
        
    def build_daylight_branch(self):
        """Build daylight CSI branch - standard preprocessing"""
        # Source: CSI camera
        src = Gst.ElementFactory.make("nvarguscamerasrc", "daylight-src")
        src.set_property("sensor-id", 2)
        
        # Caps: 1920x1080@30
        caps = Gst.Caps.from_string(
            "video/x-raw(memory:NVMM), width=1920, height=1080, "
            "format=NV12, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "daylight-caps")
        caps_filter.set_property("caps", caps)
        
        # Flip upside down using nvvideoconvert
        flip = Gst.ElementFactory.make("nvvideoconvert", "daylight-flip")
        flip.set_property("flip-method", 2)  # 2 = 180 degrees
        
        # Stream muxer
        mux = Gst.ElementFactory.make("nvstreammux", "daylight-mux")
        mux.set_property("width", 1920)
        mux.set_property("height", 1080)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)
        
        # Standard nvinfer (uses thermal batch-1 model for now)
        # TODO: Create batch-1 daylight model for production
        infer = Gst.ElementFactory.make("nvinfer", "daylight-infer")
        infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
        infer.set_property("gpu-id", 0)
        infer.set_property("input-tensor-meta", False)  # Standard preprocessing
        
        # OSD
        osd = Gst.ElementFactory.make("nvdsosd", "daylight-osd")
        
        # FPS probe
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.daylight_fps_probe, None)
        
        # Sink
        sink = Gst.ElementFactory.make("nveglglessink", "daylight-sink")
        sink.set_property("sync", False)
        
        # Add elements
        elements = [src, caps_filter, flip, mux, infer, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)
        
        # Link elements
        src.link(caps_filter)
        caps_filter.link(flip)
        
        src_pad = flip.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)
        
        mux.link(infer)
        infer.link(osd)
        osd.link(sink)
        
        return True
    
    def build_thermal_branch(self):
        """Build thermal USB branch"""
        # Source: USB camera (thermal)
        src = Gst.ElementFactory.make("v4l2src", "thermal-src")
        src.set_property("device", "/dev/video0")

        caps = Gst.Caps.from_string(
            "video/x-raw, width=640, height=512, format=YUY2, framerate=30/1"
        )
        caps_filter = Gst.ElementFactory.make("capsfilter", "thermal-caps")
        caps_filter.set_property("caps", caps)

        convert = Gst.ElementFactory.make("videoconvert", "thermal-convert")
        queue = Gst.ElementFactory.make("queue", "thermal-queue")
        queue.set_property("max-size-buffers", 3)

        nvconvert = Gst.ElementFactory.make("nvvideoconvert", "thermal-nvconvert")
        nvconvert.set_property("nvbuf-memory-type", 0)

        nvmm_caps = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12")
        nvmm_filter = Gst.ElementFactory.make("capsfilter", "thermal-nvmm-caps")
        nvmm_filter.set_property("caps", nvmm_caps)
        
        mux = Gst.ElementFactory.make("nvstreammux", "thermal-mux")
        mux.set_property("width", 640)
        mux.set_property("height", 512)
        mux.set_property("batch-size", 1)
        mux.set_property("live-source", True)

        infer = Gst.ElementFactory.make("nvinfer", "thermal-infer")
        infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
        infer.set_property("gpu-id", 0)
        infer.set_property("input-tensor-meta", False)

        osd = Gst.ElementFactory.make("nvdsosd", "thermal-osd")
        
        osd_sink_pad = osd.get_static_pad("sink")
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.thermal_fps_probe, None)

        sink = Gst.ElementFactory.make("nveglglessink", "thermal-sink")
        sink.set_property("sync", False)

        elements = [src, caps_filter, convert, queue, nvconvert, nvmm_filter, mux, infer, osd, sink]
        for elem in elements:
            self.pipeline.add(elem)

        # Link
        src.link(caps_filter)
        caps_filter.link(convert)
        convert.link(queue)
        queue.link(nvconvert)
        nvconvert.link(nvmm_filter)

        src_pad = nvmm_filter.get_static_pad("src")
        sink_pad = mux.get_request_pad("sink_0")
        src_pad.link(sink_pad)

        mux.link(infer)
        infer.link(osd)
        osd.link(sink)

        return True
    
    def daylight_fps_probe(self, pad, info, user_data):
        if self.daylight_start_time is None:
            self.daylight_start_time = time.time()
        self.daylight_frame_count += 1
        self.print_fps()
        return Gst.PadProbeReturn.OK
    
    def thermal_fps_probe(self, pad, info, user_data):
        if self.thermal_start_time is None:
            self.thermal_start_time = time.time()
        self.thermal_frame_count += 1
        return Gst.PadProbeReturn.OK
    
    def print_fps(self):
        if self.daylight_frame_count % 100 == 0:
            daylight_elapsed = time.time() - (self.daylight_start_time or time.time())
            thermal_elapsed = time.time() - (self.thermal_start_time or time.time())
            
            daylight_fps = self.daylight_frame_count / max(0.1, daylight_elapsed)
            thermal_fps = self.thermal_frame_count / max(0.1, thermal_elapsed) if self.thermal_start_time else 0
            
            print(f"Daylight: {daylight_fps:.1f} FPS | Thermal: {thermal_fps:.1f} FPS | Running: {daylight_elapsed:.0f}s")
    
    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End of stream")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            loop.quit()
        return True
    
    def run(self):
        print("Building daylight branch...")
        if not self.build_daylight_branch():
            return False
            
        print("Building thermal branch...")
        if not self.build_thermal_branch():
            return False
        
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)
        
        print("Starting pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        self.pipeline.set_state(Gst.State.NULL)
        return True

if __name__ == "__main__":
    pipeline = DualCameraPipeline()
    pipeline.run()
