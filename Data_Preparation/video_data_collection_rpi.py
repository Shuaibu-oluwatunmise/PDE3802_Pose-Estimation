#!/usr/bin/env python3
"""
RPi Camera Data Collection Tool
Same usage as video_data_collection.py but uses RPi CSI Camera via GStreamer.
"""

import os
import time
from datetime import datetime
import cv2
import numpy as np

# GStreamer / Env Setup
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

def format_class_name(class_input: str) -> str:
    """ "cards box" -> "CardsBox" """
    parts = class_input.strip().split()
    return "".join(p.capitalize() for p in parts)

def create_directory(class_name):
    """ data/raw_videos/<ClassName> """
    dir_path = os.path.join("data", "raw_videos", class_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_video_filename(dir_path):
    """ video_YYYYMMDD_HHMMSS.avi """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{timestamp}.avi"
    return os.path.join(dir_path, filename)

class RPiCamera:
    """Wrapper for GStreamer Pipeline"""
    def __init__(self, width=640, height=480, fps=30):
        Gst.init(None)
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.sink = None
        
        gst_str = (
            "libcamerasrc ! "
            f"video/x-raw,width={width},height={height},format=NV12,framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        
        try:
            self.pipeline = Gst.parse_launch(gst_str)
            self.sink = self.pipeline.get_by_name("sink")
        except Exception as e:
            print(f"Error creating pipeline: {e}")
            self.pipeline = None
            
    def start(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.PLAYING)
            # Warmup
            time.sleep(1.0)
            return True
        return False

    def read_frame(self):
        if not self.sink:
            return None
        
        sample = self.sink.emit("try-pull-sample", 50000000) # 50ms timeout
        if not sample:
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
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)


def record_video(class_name, duration=20):
    dir_path = create_directory(class_name)
    video_path = get_video_filename(dir_path)
    
    print("Initializing RPi Camera...")
    cam = RPiCamera(width=640, height=480, fps=30)
    if not cam.start():
        print("Failed to start camera.")
        return False
        
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Match camera res
    out = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
    
    print(f"\nRecording for {duration} seconds...")
    print("Press 'q' to stop recording early.")
    
    start_time = time.time()
    
    try:
        while True:
            frame = cam.read_frame()
            if frame is None:
                continue
                
            out.write(frame)
            
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            display = frame.copy()
            cv2.putText(display, f"REC: {remaining:.1f}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display, f"Class: {class_name}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
            cv2.imshow("RPi Recording", display)
            
            if elapsed >= duration or (cv2.waitKey(1) & 0xFF == ord('q')):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        out.release()
        cv2.destroyAllWindows()
        
    print(f"Video saved to: {video_path}")
    return True

def main():
    print("=" * 50)
    print("RPi Video Data Collection Tool (CSI Camera)")
    print("=" * 50)
    print("Type 'exit' to quit.")
    
    while True:
        class_input = input("\nEnter class name: ").strip()
        
        if class_input.lower() in ["exit", "quit", "q"]:
            break
            
        if not class_input:
            print("Invalid name.")
            continue
            
        class_name = format_class_name(class_input)
        
        try:
            dur_input = input("Duration (default 20s): ").strip()
            duration = int(dur_input) if dur_input else 20
        except:
            duration = 20
            
        success = record_video(class_name, duration)
        
        if success:
            print(f"✓ Saved {class_name}")
        else:
            print("Failed.")

if __name__ == "__main__":
    main()
