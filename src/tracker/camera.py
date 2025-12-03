import os
import gi
import numpy as np
import cv2

# RPI CSI Requirement: GI requirements
gi.require_version("Gst", "1.0")
from gi.repository import Gst

class CSICamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.sink = None
        
        # Set environment variables for GStreamer
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")                

    def start(self):
        """Starts the GStreamer/libcamera pipeline."""
        Gst.init(None)
        
        gst_str = (
            f"libcamerasrc ! "
            f"video/x-raw,width={self.width},height={self.height},format=NV12,framerate={self.fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )

        try:
            self.pipeline = Gst.parse_launch(gst_str)
            self.sink = self.pipeline.get_by_name("sink")
            self.pipeline.set_state(Gst.State.PLAYING)
            print(f"✓ CSI Camera STARTED: {self.width}x{self.height} @ {self.fps}fps")
            return True
        except Exception as e:
            print(f"ERROR: Cannot start GStreamer: {e}")
            return False

    def pull_frame(self, timeout_ns=10_000_000):                
        """Grab one BGR frame from appsink."""
        if self.sink is None:                
            return None                
            
        sample = self.sink.emit("try-pull-sample", timeout_ns)                
        if sample is None:                
            return None                
            
        buf = sample.get_buffer()                
        caps = sample.get_caps().get_structure(0)                
        w = caps.get_value("width")                
        h = caps.get_value("height")                
        
        ok, mapinfo = buf.map(Gst.MapFlags.READ)                
        if not ok:                
            return None                
            
        try:                
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3)                
            return frame.copy() 
        finally:                
            buf.unmap(mapinfo)

    def stop(self):
        """Stops the pipeline."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            print("✓ CSI Camera STOPPED")