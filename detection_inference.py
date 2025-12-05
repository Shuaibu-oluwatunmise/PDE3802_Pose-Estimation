#!/usr/bin/env python3
"""
Simplified Adaptive YOLO Detection
Stable frame skipping for Raspberry Pi
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np
from ultralytics import YOLO
import time

class StableYOLODetector:
    def __init__(self, model_path):
        print("\n" + "="*60)
        print("STABLE ADAPTIVE YOLO DETECTION")
        print("="*60)
        
        # Load model
        print(f"Loading: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()
        
        # Try FP16
        try:
            self.model.model.half()
            print("✓ FP16 enabled")
        except:
            print("⚠ FP16 unavailable")
        
        self.class_names = self.model.names
        print(f"Classes: {list(self.class_names.values())}")
        
        # Camera
        self.pipeline = None
        self.sink = None
        
        # Adaptive settings
        self.frame_idx = 0
        self.skip_tracking = 3    # Skip every 3rd frame when tracking
        self.skip_search = 1      # No skip when searching
        self.no_det_count = 0
        
        # Detection cache - store as simple dict
        self.cached_boxes = []  # List of (class_name, bbox, conf)
        
        # Settings
        self.conf = 0.3
        self.iou = 0.5
        self.imgsz = 384
        
        # Colors
        self.colors = {
            'card_game': (0, 255, 0),
            'circuit_board': (255, 0, 0),
            'estop': (0, 0, 255),
            'notebook': (255, 255, 0),
            'phone': (255, 0, 255),
        }
        
        print(f"Settings: imgsz={self.imgsz}, conf={self.conf}")
        print("="*60 + "\n")
    
    def start_camera(self):
        print("Starting camera...")
        Gst.init(None)
        gst_str = (
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=2 drop=true"
        )
        self.pipeline = Gst.parse_launch(gst_str)
        self.sink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)
        print("✓ Camera ready\n")
        return True
    
    def pull_frame(self):
        if not self.sink:
            return None
        sample = self.sink.emit("try-pull-sample", 10_000_000)
        if not sample:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            return np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3).copy()
        finally:
            buf.unmap(mapinfo)
    
    def detect_and_cache(self, frame):
        """Run detection and cache results"""
        results = self.model(
            frame,
            verbose=False,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            half=True
        )
        
        result = results[0]
        self.cached_boxes = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, "unknown")
                
                self.cached_boxes.append((cls_name, (x1, y1, x2, y2), conf))
    
    def draw_cached_boxes(self, frame):
        """Draw cached detections on frame"""
        for cls_name, (x1, y1, x2, y2), conf in self.cached_boxes:
            # Get color
            color = self.colors.get(cls_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-h-10), (x1+w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        print("Controls: 'q'=quit, 's'=save\n")
        
        if not self.start_camera():
            return
        
        fps, count, last_t = 0, 0, time.time()
        saved = 0
        
        try:
            while True:
                frame = self.pull_frame()
                if frame is None:
                    continue
                
                count += 1
                
                # FPS
                now = time.time()
                if now - last_t >= 1.0:
                    fps = count
                    count = 0
                    last_t = now
                
                # Adaptive frame skip
                self.frame_idx += 1
                
                has_detections = len(self.cached_boxes) > 0
                skip_rate = self.skip_tracking if has_detections else self.skip_search
                
                run_yolo = (self.frame_idx % skip_rate == 0)
                
                # Safety: force detection if lost for 10 frames
                if not has_detections:
                    self.no_det_count += 1
                    if self.no_det_count > 10:
                        run_yolo = True
                        self.frame_idx = 0
                else:
                    self.no_det_count = 0
                
                # Run or skip
                if run_yolo:
                    self.detect_and_cache(frame)
                
                # Always draw cached boxes
                display = self.draw_cached_boxes(frame.copy())
                
                # Info overlay
                cv2.putText(display, f"FPS: {fps}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                mode = "SEARCH" if run_yolo else f"TRACK (skip {skip_rate})"
                color = (0, 255, 255) if run_yolo else (0, 255, 0)
                cv2.putText(display, mode, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(display, f"Objects: {len(self.cached_boxes)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Stable Detection', display)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"detect_{saved:03d}.jpg", display)
                    print(f"✓ Saved detect_{saved:03d}.jpg")
                    saved += 1
        
        except KeyboardInterrupt:
            pass
        finally:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
            print("\nDone!")


if __name__ == "__main__":
    StableYOLODetector('runs/detect/yolov8n_detect_V2/weights/best.pt').run()