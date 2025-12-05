#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED Live Inference - Raspberry Pi CSI Camera
Maximum FPS configuration with aggressive optimizations
"""

import os
import time

# CPU / threading optimizations
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if "DISPLAY" not in os.environ:
    print("WARN: No DISPLAY variable found. Defaulting to physical display :0")
    os.environ["DISPLAY"] = ":0"

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
#              ULTRA-PERFORMANCE CONFIGURATION
# ============================================================

MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
CONF_THRESHOLD = 0.6  # Higher threshold = fewer detections = faster

# Network inference size (lower = MUCH faster)
INFER_IMGSZ = 320  # Try: 320 (good balance) or 256 (maximum speed)

# Frame skipping (higher = faster but less responsive)
SKIP_EVERY_N_FRAMES = 2  # Process every 2nd frame (15 FPS inference from 30 FPS camera)

# Camera resolution (lower = faster)
CAMERA_WIDTH = 320   # Reduced from 640 for speed
CAMERA_HEIGHT = 240  # Reduced from 480 for speed

# Display optimizations
SHOW_SIMPLE_BOXES = True  # Use simple boxes instead of YOLO's plot() for speed
DISPLAY_SCALE = 1.0       # Scale display (>1.0 to enlarge small resolution)

# ============================================================


class UltraOptimizedInference:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.pipeline = None
        self.sink = None
        self.last_boxes = []  # Store last detection boxes for frame skipping

        print("Loading and optimizing model...")
        self.model = YOLO(model_path)

        # Fuse model layers for speed
        try:
            self.model.fuse()
            print("✓ Model fused")
        except Exception as e:
            print(f"Model fuse skipped: {e}")

        self.class_names = self.model.names
        self.colors = self._generate_colors()
        print(f"✓ Model loaded! Classes: {self.class_names}")

    def _generate_colors(self):
        """Generate consistent colors for each class"""
        np.random.seed(42)
        return {i: tuple(map(int, np.random.randint(0, 255, 3))) 
                for i in range(len(self.class_names))}

    def start_camera(self):
        """Start camera with optimized pipeline"""
        print(f"Starting camera with ultra-optimized pipeline...")
        Gst.init(None)

        gst_str = (
            "libcamerasrc ! "
            f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        try:
            self.pipeline = Gst.parse_launch(gst_str)
            self.sink = self.pipeline.get_by_name("sink")
            self.pipeline.set_state(Gst.State.PLAYING)
            print(f"✓ Camera started: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def get_frame(self):
        """Pull frame from pipeline"""
        sample = self.sink.emit("pull-sample")
        if sample is None:
            return None

        buf = sample.get_buffer()
        caps = sample.get_caps()

        height = caps.get_structure(0).get_value("height")
        width = caps.get_structure(0).get_value("width")

        result, mapinfo = buf.map(Gst.MapFlags.READ)
        if not result:
            return None

        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=mapinfo.data
        )

        buf.unmap(mapinfo)
        return frame

    def draw_simple_boxes(self, frame, boxes):
        """Fast box drawing (cheaper than YOLO's plot())"""
        for box in boxes:
            x1, y1, x2, y2 = map(int, box['xyxy'])
            cls = box['cls']
            conf = box['conf']
            
            color = self.colors[cls]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[cls]}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def run(self):
        """Main inference loop with all optimizations"""
        if not self.start_camera():
            return

        print("\n" + "="*60)
        print("ULTRA-OPTIMIZED MODE")
        print(f"Inference size: {INFER_IMGSZ}x{INFER_IMGSZ}")
        print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"Frame skip: Every {SKIP_EVERY_N_FRAMES} frames")
        print(f"Simple boxes: {SHOW_SIMPLE_BOXES}")
        print("\nPress 'q' to quit | 's' to save frame")
        print("="*60 + "\n")

        frame_idx = 0
        inference_count = 0
        last_time = time.time()
        fps = 0.0
        inf_fps = 0.0
        avg_inf_time = 0.0

        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    break

                frame_idx += 1

                # FPS calculation
                now = time.time()
                if now - last_time >= 1.0:
                    fps = frame_idx / (now - last_time)
                    inf_fps = inference_count / (now - last_time)
                    frame_idx = 0
                    inference_count = 0
                    last_time = now

                # Determine if we should run inference
                should_infer = (SKIP_EVERY_N_FRAMES == 1) or \
                              (frame_idx % SKIP_EVERY_N_FRAMES == 0)

                if should_infer:
                    # Run inference
                    t0 = time.time()
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        imgsz=INFER_IMGSZ,
                        verbose=False
                    )
                    inf_time = (time.time() - t0) * 1000
                    
                    avg_inf_time = 0.9 * avg_inf_time + 0.1 * inf_time
                    inference_count += 1

                    # Extract boxes for simple drawing
                    boxes_data = results[0].boxes
                    self.last_boxes = []
                    for box in boxes_data:
                        self.last_boxes.append({
                            'xyxy': box.xyxy[0].cpu().numpy(),
                            'conf': float(box.conf[0]),
                            'cls': int(box.cls[0])
                        })

                # Draw detections
                if SHOW_SIMPLE_BOXES:
                    display_frame = frame.copy()
                    if self.last_boxes:
                        display_frame = self.draw_simple_boxes(display_frame, self.last_boxes)
                else:
                    if should_infer:
                        display_frame = results[0].plot()
                    else:
                        display_frame = frame.copy()

                # Scale display if needed
                if DISPLAY_SCALE != 1.0:
                    new_width = int(display_frame.shape[1] * DISPLAY_SCALE)
                    new_height = int(display_frame.shape[0] * DISPLAY_SCALE)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))

                # Minimal overlay (faster than fancy overlay)
                num_det = len(self.last_boxes)
                info_text = f"FPS:{fps:.0f} | Inf:{inf_fps:.0f} ({avg_inf_time:.0f}ms) | Det:{num_det}"
                cv2.putText(display_frame, info_text, (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('ULTRA-OPTIMIZED', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    fname = f'ultra_opt_{int(time.time())}.jpg'
                    cv2.imwrite(fname, display_frame)
                    print(f"✓ Saved: {fname}")

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
            print(f"\n{'='*60}")
            print(f"Final stats: {fps:.1f} FPS | Avg inference: {avg_inf_time:.1f}ms")
            print(f"{'='*60}")


def main():
    inference = UltraOptimizedInference(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD
    )
    inference.run()


if __name__ == '__main__':
    main()