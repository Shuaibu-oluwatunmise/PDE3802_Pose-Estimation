#!/usr/bin/env python3
"""
Simple Live Inference Script - Raspberry Pi CSI Camera Version
Optimised for higher FPS on Raspberry Pi
"""

import os
import time

# ---------- CPU / threading env tweaks (helps a bit on small CPUs) ----------
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

# *** IMPORTANT: smaller imgsz = much faster on RPi ***
INFER_IMGSZ = 320        # Try 320 or even 256 for speed
SKIP_EVERY_N_FRAMES = 1  # 0 = no skipping, 1 = infer every frame, 2 = infer every other frame, etc.


class RpiCameraInference:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.pipeline = None
        self.sink = None

        # Load YOLO model
        print("Loading model...")
        self.model = YOLO(model_path)

        # Fuse model for faster inference (conv+bn)
        try:
            self.model.fuse()
            print("Model fused for faster inference.")
        except Exception as e:
            print(f"Model fuse skipped: {e}")

        print(f"Model loaded! Classes: {self.model.names}")

    def start_camera(self):
        """Start Raspberry Pi CSI Camera via GStreamer/libcamera"""
        print(f"Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
        Gst.init(None)

        # NOTE: You can also reduce capture resolution here to speed things up even more
        # e.g. width=320,height=240
        gst_str = (
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            # More real-time friendly appsink config:
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
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
        """Run live inference"""
        if not self.start_camera():
            print("Failed to start camera!")
            return

        print("\nRunning inference... Press 'q' to quit")

        frame_count = 0
        last_time = time.time()
        fps = 0.0

        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    print("Failed to get frame")
                    break

                frame_count += 1

                # --- Measure FPS (overall loop, including inference & drawing) ---
                now = time.time()
                if now - last_time >= 1.0:
                    fps = frame_count / (now - last_time)
                    last_time = now
                    frame_count = 0

                # --- OPTIONAL: skip frames to gain speed ---
                # If SKIP_EVERY_N_FRAMES > 1, only run inference on every Nth frame
                if SKIP_EVERY_N_FRAMES > 1 and (frame_count % SKIP_EVERY_N_FRAMES != 0):
                    # Just show the raw frame without inference
                    display_frame = frame.copy()
                else:
                    # --- Run inference with smaller network input size ---
                    # Ultralytics internally resizes to imgsz, so this is cheap.
                    results = self.model(
                        frame,
                        conf=self.conf_threshold,
                        imgsz=INFER_IMGSZ,  # <<< HUGE for speed
                        verbose=False
                    )

                    # Get annotated frame
                    # (This plotting is also not free. If FPS is still bad, you can
                    #  draw minimal boxes yourself instead of using .plot().)
                    display_frame = results[0].plot()

                # Put FPS text on screen (for debugging performance)
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow('YOLO Live Detection - RPi Camera', display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
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
