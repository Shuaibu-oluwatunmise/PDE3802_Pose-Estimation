#!/usr/bin/env python3
"""
Simple Live Inference Script - Raspberry Pi CSI Camera Version
Quick start for real-time detection with RPi Camera Module
"""

import os
# RPI CSI: Set environment variables
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")

if "DISPLAY" not in os.environ:
    print("WARN: No DISPLAY variable found. Defaulting to physical display :0")
    os.environ["DISPLAY"] = ":0"

import gi
# RPI CSI: GI requirements
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
CONF_THRESHOLD = 0.5

class RpiCameraInference:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.pipeline = None
        self.sink = None
        
        # Load YOLO model
        print("Loading model...")
        self.model = YOLO(model_path)
        print(f"Model loaded! Classes: {self.model.names}")
    
    def start_camera(self):
        """Start Raspberry Pi CSI Camera via GStreamer/libcamera"""
        print(f"Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
        Gst.init(None)
        
        gst_str = (
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )
        
        try:
            self.pipeline = Gst.parse_launch(gst_str)
            self.sink = self.pipeline.get_by_name("sink")
            self.pipeline.set_state(Gst.State.PLAYING)
            print(f"✓ GStreamer pipeline STARTED: 640x480")
            return True
        except Exception as e:
            print(f"ERROR: Cannot start GStreamer pipeline: {e}")
            return False
    
    def get_frame(self):
        """Pull frame from GStreamer pipeline"""
        sample = self.sink.emit("pull-sample")
        if sample is None:
            return None
        
        buf = sample.get_buffer()
        caps = sample.get_caps()
        
        # Get frame dimensions
        height = caps.get_structure(0).get_value("height")
        width = caps.get_structure(0).get_value("width")
        
        # Extract buffer data
        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if not result:
            return None
        
        # Convert to numpy array
        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=mapinfo.data
        )
        
        buf.unmap(mapinfo)
        return frame
    
    def run(self):
        """Run live inference"""
        # Start camera
        if not self.start_camera():
            print("Failed to start camera!")
            return
        
        print("\nRunning inference... Press 'q' to quit")
        
        try:
            while True:
                # Get frame from pipeline
                frame = self.get_frame()
                
                if frame is None:
                    print("Failed to get frame")
                    break
                
                # Run inference
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Display
                cv2.imshow('YOLO Live Detection - RPi Camera', annotated_frame)
                
                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
            print("Camera stopped")


def main():
    inference = RpiCameraInference(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD
    )
    inference.run()


if __name__ == '__main__':
    main()