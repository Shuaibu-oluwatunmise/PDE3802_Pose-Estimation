#!/usr/bin/env python3
"""
Complete 5-Object 6DOF Pose Tracker with ROS2 TF Publishing
OPTIMIZED FOR RASPBERRY PI 4

Changes from hybrid_rpi3.py:
- Threaded Video Capture
- ROI-based Feature Extraction (Significant Speedup)
- Reduced Feature Count
- TFLite Model Support
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
import threading
import time
import queue

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations

from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ============================================================================
# CONFIGURATION
# ============================================================================
# Try to find optimized models first
YOLO_MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
if os.path.exists('runs/detect/yolov8n_detect_V2/weights/best.tflite'):
    YOLO_MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.tflite'
elif os.path.exists('runs/detect/yolov8n_detect_V2/weights/best.onnx'):
    YOLO_MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.onnx'

# ArUco Configuration
ARUCO_DICT = cv2.aruco.DICT_4X4_50
ARUCO_IDS = {"phone": 0, "estop": 1}
ARUCO_MARKER_SIZES = {"phone": 0.0365, "estop": 0.0365}

# Object Specifications
OBJECT_SPECS = {
    "card_game": {"width_m": 0.093, "height_m": 0.115, "color": (0, 255, 0), "method": "feature"},
    "circuit_board": {"width_m": 0.210, "height_m": 0.210, "color": (255, 0, 0), "method": "feature"},
    "notebook": {"width_m": 0.147, "height_m": 0.209, "color": (0, 0, 255), "method": "feature"},
    "phone": {"color": (255, 255, 0), "method": "aruco", "aruco_id": ARUCO_IDS["phone"], "marker_size_m": ARUCO_MARKER_SIZES["phone"]},
    "estop": {"color": (255, 0, 255), "method": "aruco", "aruco_id": ARUCO_IDS["estop"], "marker_size_m": ARUCO_MARKER_SIZES["estop"]},
}

MIN_DETECTION_CONFIDENCE = 0.60 # Slightly lower for speed/recall tradeoff
STABLE_FRAMES_NEEDED = 3        # Reduced for responsiveness
MIN_FEATURES = 15
ORB_FEATURES = 1000             # Reduced from 2000 for speed
MIN_MATCH_COUNT = 10            # Reduced slightly
MIN_PNP_POINTS = 6
MIN_PNP_INLIERS = 6
REPROJ_ERROR_THRESH = 4.0

DRAW_BBOX = True
DRAW_INLIERS = True
UNDISTORT = False
CAMERA_FRAME = "camera_link_G4"

# ============================================================================
# UTILITIES
# ============================================================================
def load_calibration():
    try:
        K = np.array(CAMERA_MATRIX, dtype=np.float32)
        dist = np.array(DIST_COEFFS, dtype=np.float32) if DIST_COEFFS is not None else None
        print("Camera calibration loaded successfully!")
        return K, dist
    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        return K, None

def bbox_iou(box1, box2):
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2
    x_left = max(x1_tl, x2_tl)
    y_top = max(y1_tl, y2_tl)
    x_right = min(x1_br, x2_br)
    y_bottom = min(y1_br, y2_br)
    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1, area2 = (x1_br - x1_tl) * (y1_br - y1_tl), (x2_br - x2_tl) * (y2_br - y2_tl)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def is_blurry(image, threshold=100):
    val = cv2.Laplacian(image, cv2.CV_64F).var()
    return val < threshold

# ============================================================================
# THREADED CAMERA
# ============================================================================
class ThreadedCamera:
    """Acquires frames in a separate thread to prevent blocking main loop"""
    def __init__(self, logger):
        self.logger = logger
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.pipeline = None
        self.sink = None
        
        Gst.init(None)
        gst_str = (
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        try:
            self.pipeline = Gst.parse_launch(gst_str)
            self.sink = self.pipeline.get_by_name("sink")
            self.sink.connect("new-sample", self._on_new_sample)
        except Exception as e:
            self.logger.error(f"GStreamer Error: {e}")
            self.pipeline = None

    def _on_new_sample(self, sink):
        if not self.running: return Gst.FlowReturn.OK
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w, h = caps.get_value("width"), caps.get_value("height")
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if ok:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3)
                with self.lock:
                    self.frame = arr.copy()
                buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def start(self):
        if self.pipeline:
            self.running = True
            self.pipeline.set_state(Gst.State.PLAYING)
            self.logger.info("Camera thread started")
            return True
        return False

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        self.logger.info("Camera thread stopped")

# ============================================================================
# FPS COUNTER
# ============================================================================
class FPSCounter:
    def __init__(self, logger):
        self.logger = logger
        self.start_time = time.time()
        self.frame_count = 0
        self.interval = 2.0 # Print every 2 seconds

    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > self.interval:
            fps = self.frame_count / elapsed
            self.logger.info(f"FPS: {fps:.2f}")
            self.start_time = time.time()
            self.frame_count = 0

# ============================================================================
# FEATURE TRACKER CLASS (OPTIMIZED)
# ============================================================================
class FeatureTracker:
    def __init__(self, obj_name, obj_specs, orb, bf_matcher, K, dist):
        self.obj_name = obj_name
        self.obj_width = obj_specs["width_m"]
        self.obj_height = obj_specs["height_m"]
        self.color = obj_specs["color"]
        self.orb = orb
        self.bf = bf_matcher
        self.K = K
        self.dist = dist
        
        self.ref_image, self.ref_kp, self.ref_des, self.plane_ref = None, None, None, None
        self.has_reference = False
        self.stable_bbox_buffer = []
        self.prev_rvec, self.prev_tvec = None, None
        self.last_bbox = None
        self.is_tracking = False

    def is_bbox_stable(self, bbox):
        self.stable_bbox_buffer.append(bbox)
        if len(self.stable_bbox_buffer) > STABLE_FRAMES_NEEDED: self.stable_bbox_buffer.pop(0)
        if len(self.stable_bbox_buffer) < STABLE_FRAMES_NEEDED: return False
        first = self.stable_bbox_buffer[0]
        return all(bbox_iou(first, b) > 0.85 for b in self.stable_bbox_buffer[1:])

    def create_reference(self, frame, bbox, logger):
        x1, y1, x2, y2 = bbox
        # Add margin to crop
        h, w = frame.shape[:2]
        m = 10
        x1, y1 = max(0, x1-m), max(0, y1-m)
        x2, y2 = min(w, x2+m), min(h, y2+m)
        
        crop = frame[y1:y2, x1:x2]
        
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if is_blurry(gray_crop):
            logger.warn(f"{self.obj_name}: Reference too blurry")
            return False
            
        kp_crop, des_crop = self.orb.detectAndCompute(gray_crop, None)
        if des_crop is None or len(kp_crop) < MIN_FEATURES:
            return False

        # Adjust KeyPoints to global coordinates and create 3D plane
        sx, sy = self.obj_width / (x2-x1), self.obj_height / (y2-y1)
        
        plane_ref, valid_kp, valid_des = [], [], []
        
        # Mapping: Center of bbox is (0,0,0).
        # We need to preserve "Y Up" orientation (Standard in many robotics/ROS simple setups).
        # Original code: Y = -v * sy (Pixel down -> Physical Up).
        # New Center-based: Y = (-v * sy) + Y_offset.
        # At v=0 (top), Y should be +Height/2.
        # At v=h (bot), Y should be -Height/2.
        
        for i, kp in enumerate(kp_crop):
            u, v = kp.pt # relative to crop
            
            # X Right is positive (Same as pixel u)
            X = (u * sx) - (self.obj_width / 2.0)
            
            # Y Up is positive (Opposite to pixel v)
            Y = (-v * sy) + (self.obj_height / 2.0)
            
            plane_ref.append([X, Y, 0.0])
            
            # Keypoint global coords
            kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
            valid_kp.append(kp)
            valid_des.append(des_crop[i])

        self.ref_kp = valid_kp
        self.ref_des = np.array(valid_des)
        self.plane_ref = np.array(plane_ref, dtype=np.float32)
        self.has_reference = True
        self.last_bbox = bbox
        logger.info(f"Reference set for {self.obj_name}: {len(valid_kp)} features")
        return True

    def predict_next_bbox(self, frame_shape):
        """Project the 3D plane back to image to estimate next bounding box"""
        if self.prev_rvec is None or self.prev_tvec is None:
            return self.last_bbox

        # Define 3D corners of the object
        w, h = self.obj_width, self.obj_height
        corners_3d = np.float32([
            [-w/2,  h/2, 0], # Top-Left
            [ w/2,  h/2, 0], # Top-Right
            [ w/2, -h/2, 0], # Bottom-Right
            [-w/2, -h/2, 0]  # Bottom-Left
        ])
        
        imgpts, _ = cv2.projectPoints(corners_3d, self.prev_rvec, self.prev_tvec, self.K, self.dist)
        imgpts = imgpts.reshape(-1, 2)
        
        # Find binding rect
        x_min, y_min = np.min(imgpts, axis=0)
        x_max, y_max = np.max(imgpts, axis=0)
        
        h_img, w_img = frame_shape[:2]
        x1 = max(0, int(x_min))
        y1 = max(0, int(y_min))
        x2 = min(w_img, int(x_max))
        y2 = min(h_img, int(y_max))
        
        # Add some margin for movement
        m = 20
        x1, y1 = max(0, x1-m), max(0, y1-m)
        x2, y2 = min(w_img, x2+m), min(h_img, y2+m)
        
        if x2 <= x1 or y2 <= y1:
            return self.last_bbox
            
        return [x1, y1, x2, y2]

    def track_pose(self, frame, bbox):
        if not self.has_reference: return None, "No Ref"
        
        self.last_bbox = bbox # Update known position
        
        # ROI TRACKING OPTIMIZATION
        # Only search in the area of the detected bbox + margin
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        m = 30 # Larger margin for tracking
        x1, y1 = max(0, x1-m), max(0, y1-m)
        x2, y2 = min(w_img, x2+m), min(h_img, y2+m)
        
        crop = frame[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        kp_live_crop, des_live = self.orb.detectAndCompute(gray_crop, None)
        if des_live is None or len(kp_live_crop) < MIN_FEATURES:
            self.is_tracking = False
            return None, "No Feat"

        # Shift keypoints back to global coordinates
        kp_live_global = []
        for kp in kp_live_crop:
            kp.pt = (kp.pt[0] + x1, kp.pt[1] + y1)
            kp_live_global.append(kp)
            
        matches = self.bf.knnMatch(self.ref_des, des_live, k=2)
        good_matches = []
        for m_match, n_match in matches:
            if m_match.distance < 0.75 * n_match.distance:
                good_matches.append(m_match)
        
        if len(good_matches) < MIN_MATCH_COUNT:
            self.is_tracking = False
            return None, f"Low Match {len(good_matches)}"
            
        obj_pts = []
        img_pts = []
        for m in good_matches:
            obj_pts.append(self.plane_ref[m.queryIdx])
            img_pts.append(kp_live_global[m.trainIdx].pt)
            
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        if len(obj_pts) < MIN_PNP_POINTS:
            self.is_tracking = False
            return None, "Low PnP Pts"

        # Use previous pose as initial guess if available
        use_guess = self.prev_rvec is not None
        rvec_init = self.prev_rvec if use_guess else None
        tvec_init = self.prev_tvec if use_guess else None

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, self.K, self.dist,
                useExtrinsicGuess=use_guess,
                rvec=rvec_init, tvec=tvec_init,
                iterationsCount=100, reprojectionError=8.0,
                confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
            )
        except:
             self.is_tracking = False
             return None, "PnP Error"

        if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
             self.is_tracking = False
             return None, "PnP Failed"
             
        # Refine
        inliers = inliers.flatten()
        obj_in = obj_pts[inliers]
        img_in = img_pts[inliers]
        
        # Calculate Error
        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, self.K, self.dist)
        err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_in, axis=1))

        if err > REPROJ_ERROR_THRESH:
             self.is_tracking = False
             return None, f"High Err {err:.1f}"

        self.prev_rvec = rvec
        self.prev_tvec = tvec
        self.is_tracking = True
        
        return {
            'rvec': rvec, 'tvec': tvec, 'inliers': img_in, 
            'reproj_error': err
        }, f"OK ({len(inliers)})"

    def reset(self):
        self.has_reference = False
        self.prev_rvec = None
        self.prev_tvec = None
        self.last_bbox = None
        self.is_tracking = False

# ============================================================================
# MAIN TRACKER
# ============================================================================
class Full5ObjectTracker(Node):
    def __init__(self):
        super().__init__('full_5_object_tracker_node')
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info(f"Loading YOLO from {YOLO_MODEL_PATH}...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')
        self.get_logger().info("✓ YOLO loaded")
        
        self.K, self.dist = load_calibration()
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.feature_trackers = {}
        for name, specs in OBJECT_SPECS.items():
            if specs["method"] == "feature":
                self.feature_trackers[name] = FeatureTracker(name, specs, self.orb, self.bf, self.K, self.dist)

        self.camera = ThreadedCamera(self.get_logger())
        self.frame_count = 0
        self.fps_counter = FPSCounter(self.get_logger())
        self.last_detections = {}

    def publish_tf(self, obj_name, rvec, tvec):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = CAMERA_FRAME
        t.child_frame_id = f"{obj_name}_frame"
        t.transform.translation.x = float(tvec[0, 0])
        t.transform.translation.y = float(tvec[1, 0])
        t.transform.translation.z = float(tvec[2, 0])
        q = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([cv2.Rodrigues(rvec)[0], [[0],[0],[0]]]), [0,0,0,1]])
        )
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_broadcaster.sendTransform(t)

    def draw_axes(self, display, obj_name, rvec, tvec):
        specs = OBJECT_SPECS[obj_name]
        length = specs["marker_size_m"]*0.7 if specs["method"] == "aruco" else min(specs["width_m"], specs["height_m"])*0.7
        pts, _ = cv2.projectPoints(np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]]), rvec, tvec, self.K, self.dist)
        pts = pts.reshape(-1, 2).astype(int)
        cv2.line(display, tuple(pts[0]), tuple(pts[1]), (0,0,255), 3)
        cv2.line(display, tuple(pts[0]), tuple(pts[2]), (0,255,0), 3)
        cv2.line(display, tuple(pts[0]), tuple(pts[3]), (255,0,0), 3)

    def run(self):
        if not self.camera.start(): return
        self.get_logger().info("Setup Complete. Starting Loop...")
        
        try:
            while rclpy.ok():
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self.frame_count += 1
                self.fps_counter.update()
                
                if UNDISTORT and self.dist is not None:
                     frame = cv2.undistort(frame, self.K, self.dist)
                
                display = frame.copy()
                
                # DYNAMIC YOLO SCHEDULING
                # Check status of feature trackers
                # If any tracker HAS reference but LOST tracking, we need to find it fast -> Fast YOLO
                # If all trackers are fine -> Slow YOLO
                
                trackers_needing_help = False
                for t in self.feature_trackers.values():
                    if t.has_reference and not t.is_tracking:
                        trackers_needing_help = True
                        break
                
                # Interval logic:
                # 5:  Aggressive search (Something is lost)
                # 30: Maintenance (Smooth sailing)
                yolo_interval = 5 if trackers_needing_help else 30
                
                run_yolo = (self.frame_count % yolo_interval == 0)
                
                if run_yolo:
                    results = self.yolo_model(frame, verbose=False)
                    self.last_detections = {}
                    for r in results:
                        for box in r.boxes:
                            name = self.yolo_model.names[int(box.cls[0])]
                            if name in OBJECT_SPECS and float(box.conf[0]) > MIN_DETECTION_CONFIDENCE:
                                self.last_detections[name] = box.xyxy[0].cpu().numpy().astype(int)
                
                # ArUco
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
                aruco_map = {}
                if ids is not None:
                    ids = ids.flatten()
                    for i, id_val in enumerate(ids):
                        aruco_map[id_val] = corners[i][0]

                y_off = 30
                
                # PROCESS OBJECTS
                for name, specs in OBJECT_SPECS.items():
                    # Get bbox from YOLO (if ran) or Prediction
                    bbox = self.last_detections.get(name)
                    
                    if specs["method"] == "feature":
                        tracker = self.feature_trackers[name]
                        
                        # If we didn't run YOLO this frame, ask tracker to predict bbox from previous pose
                        if not run_yolo and tracker.has_reference and tracker.prev_rvec is not None:
                            pred_bbox = tracker.predict_next_bbox(frame.shape)
                            if pred_bbox is not None:
                                bbox = pred_bbox
                        
                        # VISUALIZE BBOX (Cyan if predicted, Green if detected via YOLO recently)
                        if bbox is not None and DRAW_BBOX:
                            # Dim color if "stale" YOLO detection and no prediction? 
                            # If run_yolo is True, it's fresh. Else it's effectively predicted/held.
                            color = specs["color"] if run_yolo else (255, 255, 0)
                            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                        if bbox is not None:
                            if not tracker.has_reference:
                                if tracker.is_bbox_stable(bbox):
                                    tracker.create_reference(frame, bbox, self.get_logger())
                                else:
                                    cv2.putText(display, f"{name}: Stabilizing", (10, y_off), 0, 0.5, (0,255,255), 1)
                            else:
                                pose, status = tracker.track_pose(frame, bbox)
                                if pose:
                                    self.publish_tf(name, pose['rvec'], pose['tvec'])
                                    self.draw_axes(display, name, pose['rvec'], pose['tvec'])
                                    t = pose['tvec'].flatten()
                                    cv2.putText(display, f"{name}: {t[0]:.2f},{t[1]:.2f},{t[2]:.2f}", (10, y_off), 0, 0.6, specs["color"], 2)
                                else:
                                    cv2.putText(display, f"{name}: {status}", (10, y_off), 0, 0.5, (0,0,255), 1)
                        else:
                             # YOLO didn't see it AND we couldn't predict it
                             # Do NOT reset reference automatically if we just lost it.
                             # Check if we should reset? Only manually on 'r'.
                             status_msg = "Searching..."
                             if tracker.has_reference:
                                 status_msg = "Lost (Ref Saved)"
                             
                             cv2.putText(display, f"{name}: {status_msg}", (10, y_off), 0, 0.5, (100,100,100), 1)
                    
                    # ARUCO TRACKING
                    elif specs["method"] == "aruco":
                        tid = specs["aruco_id"]
                        if tid in aruco_map:
                            c = aruco_map[tid]
                            # SolvePnP
                            ms = specs["marker_size_m"]
                            op = np.array([[-ms/2, ms/2, 0], [ms/2, ms/2, 0], [ms/2, -ms/2, 0], [-ms/2, -ms/2, 0]], dtype=np.float32)
                            ret, rv, tv = cv2.solvePnP(op, c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                            if ret:
                                self.publish_tf(name, rv, tv)
                                self.draw_axes(display, name, rv, tv)
                                t = tv.flatten()
                                cv2.putText(display, f"{name}: {t[0]:.2f},{t[1]:.2f},{t[2]:.2f}", (10, y_off), 0, 0.6, specs["color"], 2)
                        
                    y_off += 20

                # FPS Calc (Running Average)
                # ... skip for brevity, or add simple one
                
                cv2.imshow("Optimized Tracker (RPi4)", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('r'): 
                    for t in self.feature_trackers.values(): t.reset()
                
                rclpy.spin_once(self, timeout_sec=0)

        except KeyboardInterrupt:
            pass
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            self.destroy_node()
            rclpy.shutdown()

def main():
    rclpy.init()
    node = Full5ObjectTracker()
    node.run()

if __name__ == "__main__":
    main()
