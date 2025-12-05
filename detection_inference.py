#!/usr/bin/env python3
"""
Simple YOLO Live Inference Script
Detects all 5 classes and displays them with bounding boxes
Perfect for testing and debugging
"""

import os
# RPI CSI: Set environment variables
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")

if "DISPLAY" not in os.environ:
    print("WARN: No DISPLAY variable found. Defaulting to physical display :0")
    os.environ["DISPLAY"] = ":0"

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np
from ultralytics import YOLO
import time

class SimpleYOLOInference:
    def __init__(self, model_path, camera_index=1):
        print("\n" + "="*60)
        print("SIMPLE YOLO LIVE INFERENCE")
        print("="*60)
        
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()
        
        self.class_names = self.model.names
        print(f"\nModel Classes: {self.class_names}")
        print(f"Number of classes: {len(self.class_names)}")
        print("="*60 + "\n")
        
        # GStreamer setup
        self.pipeline = None
        self.sink = None
        self.camera_index = camera_index
        
        # Detection settings
        self.conf_threshold = 0.3  # Confidence threshold
        self.iou_threshold = 0.5   # NMS IOU threshold
        self.img_size = 384        # Inference size
        
        # Colors for each class (BGR format)
        self.colors = [
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (128, 0, 128),    # Purple
            (255, 165, 0),    # Orange
        ]
    
    def start_camera(self):
        """Start Raspberry Pi CSI Camera"""
        print("Starting Raspberry Pi CSI Camera...")
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
            print("✓ Camera started: 640x480\n")
            return True
        except Exception as e:
            print(f"✗ Camera error: {e}")
            return False
    
    def pull_frame(self, timeout_ns=10_000_000):
        """Read frame from camera"""
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
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return frame, 0
        
        detection_count = 0
        
        for box in result.boxes:
            # Get box info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names.get(class_id, "unknown")
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Get label size for background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            detection_count += 1
            
            # Print to console
            print(f"  └─ {class_name} (ID={class_id}): {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
        
        return frame, detection_count
    
    def run(self):
        """Main inference loop"""
        print("Controls:")
        print("  'q' or ESC - Quit")
        print("  's' - Save frame")
        print("  '+' - Increase confidence threshold")
        print("  '-' - Decrease confidence threshold")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return
        
        frame_count = 0
        saved_count = 0
        fps = 0
        last_time = time.time()
        
        print("Starting inference...\n")
        
        try:
            while True:
                # Get frame
                frame = self.pull_frame()
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # Run YOLO inference
                results = self.model(
                    frame,
                    verbose=False,
                    imgsz=self.img_size,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold
                )
                
                # Draw detections
                print(f"\rFrame detections:", end="")
                display_frame, det_count = self.draw_detections(frame.copy(), results)
                
                if det_count == 0:
                    print(" None")
                else:
                    print(f" {det_count} objects")
                
                # Draw info overlay
                info_y = 30
                
                # FPS
                cv2.putText(
                    display_frame,
                    f"FPS: {fps}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                info_y += 30
                
                # Confidence threshold
                cv2.putText(
                    display_frame,
                    f"Conf Threshold: {self.conf_threshold:.2f}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )
                info_y += 30
                
                # Detection count
                cv2.putText(
                    display_frame,
                    f"Detections: {det_count}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Show frame
                cv2.imshow('YOLO Live Inference', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    print("\n\nExiting...")
                    break
                
                elif key == ord('s'):  # Save frame
                    filename = f"yolo_detection_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"\n✓ Saved: {filename}")
                    saved_count += 1
                
                elif key == ord('+') or key == ord('='):  # Increase threshold
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"\n✓ Confidence threshold: {self.conf_threshold:.2f}")
                
                elif key == ord('-') or key == ord('_'):  # Decrease threshold
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"\n✓ Confidence threshold: {self.conf_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
            print("\nCleaned up. Goodbye!")


def main():
    # Configuration
    model_path = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
    camera_index = 1
    
    # Create and run inference
    detector = SimpleYOLOInference(model_path, camera_index)
    detector.run()


if __name__ == "__main__":
    main()