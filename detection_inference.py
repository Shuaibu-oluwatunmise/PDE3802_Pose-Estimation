#!/usr/bin/env python3
"""
Optimized YOLO Detection Script with Adaptive Frame Skipping
Performance optimized for Raspberry Pi
Detection only - no pose estimation
"""

import os
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

class AdaptiveYOLODetector:
    def __init__(self, model_path):
        print("\n" + "="*60)
        print("ADAPTIVE YOLO DETECTION (OPTIMIZED FOR PI)")
        print("="*60)
        
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.model.fuse()
        
        # Try FP16 for speed
        try:
            self.model.model.half()
            print("✓ FP16 (half precision) enabled")
        except:
            print("⚠ FP16 not available, using FP32")
        
        self.class_names = self.model.names
        print(f"\nModel Classes: {self.class_names}")
        print(f"Number of classes: {len(self.class_names)}")
        print("="*60 + "\n")
        
        # Camera
        self.pipeline = None
        self.sink = None
        
        # ADAPTIVE FRAME SKIPPING (like your working script)
        self.frame_idx = 0
        self.yolo_every_n_tracking = 3    # Run YOLO every 3rd frame when tracking
        self.yolo_every_n_search = 1      # Run YOLO every frame when searching
        self.no_det_frames = 0            # Counter for lost detections
        
        # Cache last detections
        self.last_bboxes = {}
        self.last_confidences = {}
        
        # Detection settings (optimized like your script)
        self.conf_threshold = 0.3    # Lower threshold to catch more
        self.iou_threshold = 0.5
        self.img_size = 384          # Good balance of speed/accuracy
        
        # Colors for each class
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
        
        print(f"Settings: imgsz={self.img_size}, conf={self.conf_threshold}")
        print(f"Adaptive skipping: every {self.yolo_every_n_search} frame (search) / every {self.yolo_every_n_tracking} frames (tracking)")
        print("="*60 + "\n")
    
    def start_camera(self):
        """Start Raspberry Pi CSI Camera via GStreamer"""
        print("Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
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
            print("✓ GStreamer pipeline STARTED: 640x480\n")
            return True
        except Exception as e:
            print(f"✗ Camera error: {e}")
            return False
    
    def pull_frame(self, timeout_ns=10_000_000):
        """Read frame from GStreamer pipeline"""
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
    
    def detect_objects(self, frame):
        """Run YOLO detection"""
        results = self.model(
            frame,
            verbose=False,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            half=True  # Use FP16 if available
        )
        
        return results[0]
    
    def draw_detections(self, frame, result):
        """Draw bounding boxes and labels"""
        if result.boxes is None or len(result.boxes) == 0:
            return frame, {}
        
        current_detections = {}
        
        for box in result.boxes:
            # Get box info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.class_names.get(class_id, "unknown")
            
            # Store detection
            if class_name not in current_detections:
                current_detections[class_name] = []
            current_detections[class_name].append({
                'bbox': [x1, y1, x2, y2],
                'conf': confidence
            })
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
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
        
        return frame, current_detections
    
    def run(self):
        """Main detection loop with adaptive frame skipping"""
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
        
        print("Starting adaptive detection...\n")
        
        try:
            while True:
                # Get frame
                frame = self.pull_frame()
                if frame is None:
                    continue
                
                frame_count += 1
                display_frame = frame.copy()
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # ADAPTIVE FRAME SKIPPING (like your working script)
                self.frame_idx += 1
                
                # Decide mode: "tracking" if we have detections, "search" if not
                any_detections = bool(self.last_bboxes)
                mode_every_n = self.yolo_every_n_tracking if any_detections else self.yolo_every_n_search
                
                # Determine if we run YOLO this frame
                run_yolo = (self.frame_idx % mode_every_n == 0)
                
                # Safety: if no detections for 10 frames, force search mode
                if not any_detections:
                    self.no_det_frames += 1
                else:
                    self.no_det_frames = 0
                
                if self.no_det_frames > 10:
                    run_yolo = True
                    self.frame_idx = 0
                
                # Run detection or use cached results
                if run_yolo:
                    result = self.detect_objects(frame)
                    display_frame, detections = self.draw_detections(display_frame, result)
                    self.last_bboxes = detections
                else:
                    # Use cached detections (just for display mode indicator)
                    detections = self.last_bboxes
                    if result is not None:
                        display_frame, _ = self.draw_detections(display_frame, result)
                
                # Count total detections
                total_count = sum(len(v) for v in detections.values())
                
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
                
                # Mode indicator (like your script)
                if run_yolo:
                    mode_text = "Mode: SEARCH (YOLO)"
                    mode_color = (0, 255, 255)  # Yellow
                else:
                    mode_text = f"Mode: TRACKING (SKIP {mode_every_n})"
                    mode_color = (0, 255, 0)  # Green
                
                cv2.putText(
                    display_frame,
                    mode_text,
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    mode_color,
                    2
                )
                info_y += 30
                
                # Confidence threshold
                cv2.putText(
                    display_frame,
                    f"Conf: {self.conf_threshold:.2f}",
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
                    f"Objects: {total_count}",
                    (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Show detections per class
                if total_count > 0:
                    status_y = display_frame.shape[0] - 20
                    status_text = " | ".join([f"{k}: {len(v)}" for k, v in detections.items()])
                    cv2.putText(
                        display_frame,
                        status_text,
                        (10, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
                
                # Display
                cv2.imshow('Adaptive YOLO Detection', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    print("\n\nExiting...")
                    break
                
                elif key == ord('s'):  # Save frame
                    filename = f"detection_{saved_count:03d}.jpg"
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
    
    # Create and run detector
    detector = AdaptiveYOLODetector(model_path)
    detector.run()


if __name__ == "__main__":
    main()