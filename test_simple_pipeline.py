#!/usr/bin/env python3
"""
Simple single-camera test without custom preprocessing
Tests if basic DeepStream pipeline is stable
"""

import sys
import time
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

# Build pipeline
pipeline = Gst.Pipeline.new("simple-test")
loop = GLib.MainLoop()

# Elements
src = Gst.ElementFactory.make("nvarguscamerasrc", "src")
src.set_property("sensor-id", 0)

caps = Gst.Caps.from_string(
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1"
)
caps_filter = Gst.ElementFactory.make("capsfilter", "caps")
caps_filter.set_property("caps", caps)

mux = Gst.ElementFactory.make("nvstreammux", "mux")
mux.set_property("width", 1920)
mux.set_property("height", 1080)
mux.set_property("batch-size", 1)
mux.set_property("live-source", True)

# Standard nvinfer without custom preprocessing
# This uses a batch-1 model and standard preprocessing
infer = Gst.ElementFactory.make("nvinfer", "infer")
infer.set_property("config-file-path", "config_infer_primary_thermal.txt")
infer.set_property("gpu-id", 0)
# Use standard preprocessing, not tensor meta
infer.set_property("input-tensor-meta", False)

osd = Gst.ElementFactory.make("nvdsosd", "osd")

sink = Gst.ElementFactory.make("fakesink", "sink")
sink.set_property("sync", False)

# Add elements
for elem in [src, caps_filter, mux, infer, osd, sink]:
    pipeline.add(elem)

# Link
src.link(caps_filter)
src_pad = caps_filter.get_static_pad("src")
sink_pad = mux.get_request_pad("sink_0")
src_pad.link(sink_pad)
mux.link(infer)
infer.link(osd)
osd.link(sink)

# FPS counter
frame_count = 0
start_time = time.time()

def probe_callback(pad, info, user_data):
    global frame_count, start_time
    frame_count += 1
    if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"Simple test: {fps:.1f} FPS, running {elapsed:.0f}s")
    return Gst.PadProbeReturn.OK

osd_pad = osd.get_static_pad("sink")
osd_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, None)

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    return True

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", bus_call, loop)

print("Starting simple test pipeline (no custom preprocessing)...")
pipeline.set_state(Gst.State.PLAYING)

try:
    loop.run()
except KeyboardInterrupt:
    print("Interrupted")

pipeline.set_state(Gst.State.NULL)
