#!/usr/bin/env python3
"""
Optimized Live Inference Script - Raspberry Pi CSI Camera Version
Optimised for higher FPS on Raspberry Pi with 320x320 inference
"""

import os
import time

# ---------- CPU / threading env tweaks (helps on small CPUs) ----------
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

if "DISPLAY" not in os.environ:
    print("WARN: No DISPLAY variable found. Defaulting to physical display :0")
    os.environ["DISPLAY"] = ":0"

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
CONF_THRESHOLD = 0.5

# *** PERFORMANCE TUNING PARAMETERS ***
INFER_IMGSZ = 320        # Network input size (320x320 for speed, can try 256 if still slow)
SKIP_EVERY_N_FRAMES = 1  # 1 = infer every frame, 2 = every other frame, etc.
CAMERA_WIDTH = 640       # Camera capture width (can reduce to 320 for more speed)
CAMERA_HEIGHT = 480      # Camera capture height (can reduce to 240 for more speed)


class RpiCameraInference:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.pipeline = None
        self.sink = None
        self.last_results = None  # Store last results for frame skipping

        # Load YOLO model
        print("Loading model...")
        self.model = YOLO(model_path)

        # Fuse model for faster inference (conv+bn fusion)
        try:
            self.model.fuse()
            print("✓ Model fused for faster inference")
        except Exception as e:
            print(f"Model fuse skipped: {e}")

        print(f"✓ Model loaded! Classes: {self.model.names}")

    def start_camera(self):
        """Start Raspberry Pi CSI Camera via GStreamer/libcamera"""
        print(f"Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
        Gst.init(None)

        # GStreamer pipeline with optimized settings
        gst_str = (
            "libcamerasrc ! "
            f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            # More real-time friendly appsink config:
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )

        try:
            self.pipeline = Gst.parse_launch(gst_str)
            self.sink = self.pipeline.get_by_name("sink")
            self.pipeline.set_state(Gst.State.PLAYING)
            print(f"✓ GStreamer pipeline STARTED: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
            print(f"✓ Inference size: {INFER_IMGSZ}x{INFER_IMGSZ}")
            if SKIP_EVERY_N_FRAMES > 1:
                print(f"✓ Frame skip enabled: processing every {SKIP_EVERY_N_FRAMES} frames")
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

    def run(self):
        """Run live inference with optimizations"""
        if not self.start_camera():
            print("Failed to start camera!")
            return

        print("\n" + "="*50)
        print("Running optimized inference...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("="*50 + "\n")

        frame_count = 0
        inference_count = 0
        last_time = time.time()
        fps = 0.0
        inference_fps = 0.0
        avg_inference_time = 0.0

        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    print("Failed to get frame")
                    break

                frame_count += 1

                # --- Measure overall FPS ---
                now = time.time()
                elapsed = now - last_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    inference_fps = inference_count / elapsed
                    last_time = now
                    frame_count = 0
                    inference_count = 0

                # --- Frame skipping logic ---
                should_infer = (SKIP_EVERY_N_FRAMES == 1) or (frame_count % SKIP_EVERY_N_FRAMES == 0)

                if should_infer:
                    # --- Run inference with smaller network input size ---
                    inference_start = time.time()
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        imgsz=INFER_IMGSZ,  # <<< KEY for speed: 320x320
                        verbose=False
                    )
                    inference_time = (time.time() - inference_start) * 1000  # ms
                    
                    # Update average inference time (simple moving average)
                    if avg_inference_time == 0:
                        avg_inference_time = inference_time
                    else:
                        avg_inference_time = 0.9 * avg_inference_time + 0.1 * inference_time
                    
                    inference_count += 1
                    self.last_results = results
                    
                    # Get annotated frame
                    display_frame = results[0].plot()
                else:
                    # Use previous results or just show raw frame
                    if self.last_results is not None:
                        display_frame = self.last_results[0].plot()
                    else:
                        display_frame = frame.copy()

                # --- Add performance overlay ---
                # Semi-transparent background for text
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (5, 5), (300, 85), (0, 0, 0), -1)
                display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                
                # FPS text
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                
                # Inference FPS text
                cv2.putText(
                    display_frame,
                    f"Inference FPS: {inference_fps:.1f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                
                # Inference time text
                cv2.putText(
                    display_frame,
                    f"Inference: {avg_inference_time:.1f}ms",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow('YOLO Live Detection - RPi Camera (Optimized)', display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'rpi_optimized_{int(time.time())}.jpg'
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Saved frame: {filename}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            if self.pipeline is not None:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()
            print("\n" + "="*50)
            print(f"Camera stopped")
            print(f"Final FPS: {fps:.1f}")
            print(f"Average inference time: {avg_inference_time:.1f}ms")
            print("="*50)


def main():
    print("\n" + "="*50)
    print("RPi YOLO Inference - Optimized Version")
    print("="*50)
    print(f"Configuration:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Confidence: {CONF_THRESHOLD}")
    print(f"  - Inference size: {INFER_IMGSZ}x{INFER_IMGSZ}")
    print(f"  - Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"  - Frame skip: every {SKIP_EVERY_N_FRAMES} frame(s)")
    print("="*50 + "\n")
    
    inference = RpiCameraInference(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD
    )
    inference.run()


if __name__ == '__main__':
    main()