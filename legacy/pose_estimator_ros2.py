#!/usr/bin/env python3
"""
Combined Pose Estimation System with ROS2 TF Broadcasting
Tracks all 5 objects using both Homography and ArUco methods
- Homography: game_box, repair_mat, notebook
- ArUco: wallet, headset
- Publishes TF transforms to ROS2
- Uses Raspberry Pi CSI Camera via GStreamer
"""

import os
# RPI CSI: Set environment variables needed for GStreamer + Python on RPi
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")

# If running headless / no DISPLAY, default to :0 so OpenCV windows can still work
if "DISPLAY" not in os.environ:
    print("WARN: No DISPLAY variable found. Defaulting to physical display :0")
    os.environ["DISPLAY"] = ":0"

import gi
# RPI CSI: GI requirements (GStreamer via PyGObject)
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import cv2
import numpy as np
from ultralytics import YOLO
import time

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations

# Camera intrinsics loaded from calibration module
from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ==============================================================================
# ARUCO CONFIG
# ==============================================================================

# Use 4x4 ArUco dictionary with 50 markers
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Logical object name -> ArUco ID
ARUCO_IDS = {
    "wallet": 0,
    "headset": 1,
}

# Physical marker sizes in millimetres
ARUCO_MARKER_SIZES = {
    "wallet": 50.0,   # mm
    "headset": 32.0,  # mm
}

# ==============================================================================
# OBJECT CONFIGS
# ==============================================================================

# Per-object configuration: which method, physical size, etc.
OBJECT_CONFIGS = {
    # Homography objects (planar objects tracked via ORB + homography)
    "repair_mat": {
        "label_substring": "repair_mat",  # substring to match YOLO class name
        "width_mm": 89.0,
        "height_mm": 144.0,
        "axis_color": (0, 255, 0),       # Green for drawing
        "method": "homography",
        "min_matches": 4,                # Minimum feature matches for homography
        "min_inlier_ratio": 0.25,        # Homography RANSAC consensus threshold
        "ema_alpha": 0.2,                # Smoothing factor for pose EMA
    },
    "game_box": {
        "label_substring": "game_box",
        "width_mm": 95.0,
        "height_mm": 124.0,
        "axis_color": (255, 0, 0),       # Blue
        "method": "homography",
        "min_matches": 10,
        "min_inlier_ratio": 0.5,
        "ema_alpha": 0.2,
    },
    "notebook": {
        "label_substring": "notebook",
        "width_mm": 210.0,
        "height_mm": 295.0,
        "axis_color": (0, 0, 255),       # Red
        "method": "homography",
        "min_matches": 10,
        "min_inlier_ratio": 0.5,
        "ema_alpha": 0.2,
    },
    # ArUco objects (pose from ArUco corners directly)
    "wallet": {
        "label_substring": "wallet",
        "axis_color": (255, 255, 0),     # Cyan
        "method": "aruco",
        "aruco_id": ARUCO_IDS["wallet"],
        "marker_size_mm": ARUCO_MARKER_SIZES["wallet"],
        "ema_alpha": 0.3,
    },
    "headset": {
        "label_substring": "headset",
        "axis_color": (255, 0, 255),     # Magenta
        "method": "aruco",
        "aruco_id": ARUCO_IDS["headset"],
        "marker_size_mm": ARUCO_MARKER_SIZES["headset"],
        "ema_alpha": 0.3,
    },
}
# ==============================================================================


class CombinedPoseEstimatorROS2(Node):
    """
    Main ROS2 node that:
    - Grabs frames from RPi CSI camera via GStreamer
    - Runs YOLO to detect objects
    - For homography targets: calibrates reference features & tracks with ORB
    - For ArUco targets: detects markers in YOLO ROI
    - Estimates 6DOF pose via solvePnP / solvePnPRansac
    - Publishes TF transforms into ROS2 TF tree
    - Renders visual overlays + pose debug info
    """
    def __init__(self, yolo_model_path, camera_index=1):
        # Initialize ROS2 node with a given name
        super().__init__('pose_estimator_node')
        
        # TF Broadcaster for publishing camera->object transforms
        self.tf_broadcaster = TransformBroadcaster(self)
        self.camera_frame = "camera_link"  # Parent TF frame for the camera
        
        # Log startup info
        self.get_logger().info("="*60)
        self.get_logger().info("ROS2 Pose Estimator Node Started")
        self.get_logger().info("="*60)
        
        # GStreamer pipeline handles
        self.pipeline = None
        self.sink = None
        self.camera_index = camera_index  # Not used in pipeline string, but kept for extensibility

        # Load YOLO model for detection
        self.get_logger().info(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        # Fuse model layers for faster inference
        self.yolo_model.fuse()
        
        # Inference image size and class names from YOLO
        self.yolo_imgsz = 384
        self.class_names = self.yolo_model.names
        self.get_logger().info(f"Model loaded with classes: {list(self.class_names.values())}")

        # ArUco dictionary and detector parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Try new cv2.aruco API, fall back to legacy detectMarkers if needed
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_aruco_api = True
            self.get_logger().info(f"✓ ArUco detector initialized (new API)")
        except AttributeError:
            self.use_new_aruco_api = False
            self.get_logger().info(f"✓ ArUco detector initialized (legacy API)")

        # Frame skipping / adaptive YOLO frequency
        self.frame_idx = 0
        self.yolo_every_n_tracking = 3  # Fewer YOLO calls once tracking is stable
        self.yolo_every_n_search = 1    # More frequent YOLO calls when searching
        self.no_det_frames = 0          # Counter for frames with no detections
        
        # Last known YOLO bounding boxes and confidences (for use when skipping YOLO)
        self.last_bboxes = {name: None for name in OBJECT_CONFIGS}
        self.last_confidences = {name: 0.0 for name in OBJECT_CONFIGS}

        # Per-object runtime state (class_id, method, calibration, pose, etc.)
        self.targets = {}

        # Build per-object state based on OBJECT_CONFIGS and YOLO classes
        for obj_name, cfg in OBJECT_CONFIGS.items():
            label_sub = cfg["label_substring"].lower()
            # Find YOLO class whose name contains the label substring
            class_id = next(
                (cid for cid, cname in self.class_names.items()
                 if label_sub in cname.lower()),
                None,
            )
            if class_id is None:
                self.get_logger().warn(f"'{obj_name}' class not found in model!")
            else:
                self.get_logger().info(f"✓ {obj_name.capitalize()} class found: ID={class_id} ({cfg['method']})")

            # Base state shared by all objects
            state = {
                "class_id": class_id,
                "method": cfg["method"],
                "axis_color": cfg["axis_color"],
                "ema_alpha": cfg.get("ema_alpha", 0.2),
                "rvec_smooth": None,   # Smoothed rotation vector
                "tvec_smooth": None,   # Smoothed translation vector
            }

            # Extra fields for homography-based objects
            if cfg["method"] == "homography":
                width_mm = cfg["width_mm"]
                height_mm = cfg["height_mm"]
                half_w = width_mm / 2.0
                half_h = height_mm / 2.0

                # Define 3D coordinates of the planar object corners in its own frame
                state.update({
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "object_points_3d": np.array([
                        [-half_w,  half_h, 0],
                        [ half_w,  half_h, 0],
                        [ half_w, -half_h, 0],
                        [-half_w, -half_h, 0],
                    ], dtype=np.float32),
                    "min_matches": cfg.get("min_matches", 10),
                    "min_inlier_ratio": cfg.get("min_inlier_ratio", 0.5),
                    "ref_image": None,          # Reference patch used by ORB
                    "ref_keypoints": None,      # KP for reference image
                    "ref_descriptors": None,    # Descriptors for reference image
                    "calibrated": False,        # Flag when homography reference is ready
                    "calib_buffer": [],         # Buffer of candidate frames for calibration
                    "calib_buffer_size": 3,     # How many steady frames needed
                })

            # Extra fields for ArUco-based objects
            elif cfg["method"] == "aruco":
                marker_size = cfg["marker_size_mm"]
                half_size = marker_size / 2.0

                # 3D coordinates of square marker corners (Z=0 plane)
                state.update({
                    "aruco_id": cfg["aruco_id"],
                    "marker_size_mm": marker_size,
                    "object_points_3d": np.array([
                        [-half_size,  half_size, 0],
                        [ half_size,  half_size, 0],
                        [ half_size, -half_size, 0],
                        [-half_size, -half_size, 0],
                    ], dtype=np.float32),
                })

            self.targets[obj_name] = state

        # Camera intrinsics/distortion used for solvePnP
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DIST_COEFFS
        self.get_logger().info("✓ Loaded camera parameters from camera_params.py")

        # ORB feature detector for homography objects
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
        # Brute-force matcher with Hamming distance for ORB descriptors
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.get_logger().info(f"✓ ORB initialized (2000 features)")

    def publish_tf(self, obj_name, rvec, tvec):
        """Publish TF transform for an object using current pose (rvec/tvec)."""
        t = TransformStamped()
        
        # Header with current timestamp and parent frame
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        # Child frame named "<object>_frame"
        t.child_frame_id = f"{obj_name}_frame"
        
        # Translation (convert mm to meters for TF)
        t.transform.translation.x = float(tvec[0, 0]) / 1000.0
        t.transform.translation.y = float(tvec[1, 0]) / 1000.0
        t.transform.translation.z = float(tvec[2, 0]) / 1000.0
        
        # Rotation (convert rotation vector to quaternion)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        # Build 4x4 transform matrix to feed quaternion_from_matrix
        quaternion = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([rotation_matrix, [[0], [0], [0]]]), [0, 0, 0, 1]])
        )
        
        # Set rotation quaternion
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        
        # Broadcast transform on TF
        self.tf_broadcaster.sendTransform(t)

    def start_camera(self):
        """Start Raspberry Pi CSI Camera via GStreamer."""
        self.get_logger().info(f"Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
        Gst.init(None)
        # GStreamer pipeline:
        # libcamerasrc -> 640x480 NV12 -> BGR -> appsink
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
            self.get_logger().info(f"✓ GStreamer pipeline STARTED: 640x480")
            return True
        except Exception as e:
            self.get_logger().error(f"Cannot start GStreamer pipeline: {e}")
            return False

    def pull_frame(self, timeout_ns=10_000_000):
        """
        Read a single frame from the GStreamer pipeline.
        Returns a BGR numpy array or None if no frame is available.
        """
        if self.sink is None:
            return None
        # Grab sample from appsink with timeout
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
        """
        Run YOLO detection on the given frame.
        Returns:
          bboxes: dict of obj_name -> [x1, y1, x2, y2] or None
          best_conf: dict of obj_name -> confidence value
        """
        results = self.yolo_model(frame, verbose=False, imgsz=self.yolo_imgsz, conf=0.3, iou=0.5)
        result = results[0]

        # Initialize per-object bounding boxes and confidences
        bboxes = {name: None for name in self.targets}
        best_conf = {name: 0.0 for name in self.targets}

        # No boxes detected
        if result.boxes is None or len(result.boxes) == 0:
            return bboxes, best_conf

        # For each detection: pick best (highest confidence) per logical object
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            for name, state in self.targets.items():
                if state["class_id"] is None:
                    continue
                # If YOLO class matches this object and is best so far, update
                if class_id == state["class_id"] and confidence > best_conf[name]:
                    bboxes[name] = [x1, y1, x2, y2]
                    best_conf[name] = confidence

        return bboxes, best_conf

    # ========== HOMOGRAPHY METHODS ==========

    def calibrate_reference(self, frame, obj_name, bbox):
        """
        Calibrate homography reference for a given object.
        This extracts ORB features from ROI and stores as reference.
        """
        state = self.targets[obj_name]
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(f"CALIBRATING: {obj_name.upper()}")
        self.get_logger().info('='*60)

        x1, y1, x2, y2 = bbox
        
        # If bbox covers entire frame, use full frame as ROI
        if x1 == 0 and y1 == 0 and x2 == frame.shape[1] and y2 == frame.shape[0]:
            roi = frame
        else:
            roi = frame[y1:y2, x1:x2]

        # Convert to grayscale for feature extraction
        state["ref_image"] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        state["ref_keypoints"], state["ref_descriptors"] = self.orb.detectAndCompute(
            state["ref_image"], None
        )

        # Require a minimum number of features for a robust reference
        if state["ref_descriptors"] is None or len(state["ref_keypoints"]) < 10:
            self.get_logger().warn("Not enough features! Retrying...")
            state["calibrated"] = False
            return False

        # Reset smoothing on calibration
        state["rvec_smooth"] = None
        state["tvec_smooth"] = None
        state["calibrated"] = True
        
        self.get_logger().info(f"✓ Detected {len(state['ref_keypoints'])} features")
        self.get_logger().info(f"✓ CALIBRATION COMPLETE for {obj_name.upper()}!")
        self.get_logger().info('='*60 + '\n')
        return True

    def should_auto_calibrate(self, obj_name, bbox, confidence, frame_shape):
        """
        Decide if conditions are good enough to auto-calibrate homography:
        - sufficiently high confidence
        - object covers enough of the image
        - not too close to borders
        - not extreme aspect ratio (avoid degenerate bboxes)
        """
        if confidence < 0.75:
            return False

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        H, W = frame_shape[:2]
        area = w * h
        frame_area = W * H

        area_ratio = area / frame_area
        if area_ratio < 0.08:
            return False
        
        # Require some margin to avoid cropping edges
        margin = 30
        if x1 < margin or y1 < margin or x2 > (W - margin) or y2 > (H - margin):
            return False
        
        # Reject overly elongated boxes
        aspect = max(w, h) / min(w, h)
        if aspect > 3.0:
            return False

        return True

    def accumulate_and_maybe_calibrate(self, frame, obj_name, bbox):
        """
        Accumulate candidate frames for calibration and trigger calibration
        when the object is steady for a number of frames.
        """
        state = self.targets[obj_name]
        x1, y1, x2, y2 = bbox
        
        # If the object moved too much, reset buffer
        if len(state["calib_buffer"]) > 0:
            last_bbox = state["calib_buffer"][-1]["bbox"]
            dx = abs((x1 + x2) / 2 - (last_bbox[0] + last_bbox[2]) / 2)
            dy = abs((y1 + y2) / 2 - (last_bbox[1] + last_bbox[3]) / 2)
            
            if dx > 20 or dy > 20:
                self.get_logger().info(f"Movement detected for {obj_name} - resetting calibration buffer")
                state["calib_buffer"].clear()
                return False
        
        # Extract ROI and compute ORB features
        roi = frame[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # Ignore frames with too few features
        if descriptors is None or len(keypoints) < 10:
            return False

        # Add current candidate to buffer
        state["calib_buffer"].append({
            "roi": roi, 
            "n_kp": len(keypoints),
            "bbox": [x1, y1, x2, y2]
        })

        # Show progress while collecting enough stable frames
        if len(state["calib_buffer"]) < state["calib_buffer_size"]:
            self.get_logger().info(
                f"Calibration buffer for {obj_name}: {len(state['calib_buffer'])}/{state['calib_buffer_size']} "
                f"(keypoints: {len(keypoints)}) - HOLD STEADY!"
            )
            return False

        # Choose best candidate (most features) and perform final calibration
        best_candidate = max(state["calib_buffer"], key=lambda c: c["n_kp"])
        roi_best = best_candidate["roi"]
        h_best, w_best = roi_best.shape[:2]
        
        success = self.calibrate_reference(roi_best, obj_name, [0, 0, w_best, h_best])
        # Clear buffer after calibration attempt
        state["calib_buffer"].clear()
        return success

    def track_object_homography(self, frame, obj_name, bbox):
        """
        Track a calibrated homography object:
        - Extract ROI around YOLO box
        - Match ORB features from reference to current ROI
        - Estimate homography and obtain object corners in image
        """
        state = self.targets[obj_name]

        # Must be calibrated and have valid reference descriptors
        if not state.get("calibrated") or state["ref_descriptors"] is None:
            return None

        min_matches = state["min_matches"]
        min_inlier_ratio = state["min_inlier_ratio"]

        x1_raw, y1_raw, x2_raw, y2_raw = bbox
        pad = 10
        h, w, _ = frame.shape
        # Expand ROI a bit around YOLO box
        x1 = max(0, x1_raw - pad)
        y1 = max(0, y1_raw - pad)
        x2 = min(w - 1, x2_raw + pad)
        y2 = min(h - 1, y2_raw + pad)

        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)

        if descriptors is None or len(keypoints) < 10:
            return None

        # Match reference descriptors against current ROI descriptors using KNN
        matches = self.bf_matcher.knnMatch(state["ref_descriptors"], descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Require minimum number of good matches
        if len(good_matches) < min_matches:
            return None

        # Collect corresponding 2D point sets
        src_pts = np.float32([state["ref_keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate homography between reference patch and current ROI
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        if H is None or mask is None:
            return None

        inliers = mask.ravel().tolist()
        inlier_ratio = sum(inliers) / len(inliers)
        
        # Require enough inliers for stable homography
        if inlier_ratio < min_inlier_ratio:
            return None

        # Reference patch corners in its own pixel coordinates
        h_ref, w_ref = state["ref_image"].shape
        ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)
        # Transform reference corners into ROI coordinates
        corners_roi = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)

        # Map ROI coordinates back to full-frame coordinates
        corners_frame = corners_roi.copy()
        corners_frame[:, 0] += x1
        corners_frame[:, 1] += y1

        return corners_frame

    # ========== ARUCO METHODS ==========

    def detect_aruco_in_roi(self, frame, bbox):
        """
        Detect ArUco markers in a padded ROI around the YOLO bounding box.
        Returns:
          corners_frame: list of corner arrays in full-frame coordinates
          ids: flat array of detected marker IDs
        """
        x1, y1, x2, y2 = bbox
        
        # Slightly enlarge ROI to accommodate marker edge cases
        pad = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Use new or legacy ArUco detection APIs depending on OpenCV version
        if self.use_new_aruco_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray_roi)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_roi, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None or len(ids) == 0:
            return None, None
        
        # Convert detected corners back to full-frame coordinates
        corners_frame = []
        for corner_set in corners:
            corner_adjusted = corner_set.copy()
            corner_adjusted[0, :, 0] += x1
            corner_adjusted[0, :, 1] += y1
            corners_frame.append(corner_adjusted)
        
        return corners_frame, ids.flatten()

    # ========== POSE ESTIMATION ==========

    def estimate_pose_pnp(self, obj_name, corners_2d):
        """
        Estimate 6DOF pose (rvec, tvec) from 2D-3D correspondences using PnP.
        - For ArUco: uses IPPE_SQUARE which is specialized for square planar markers
        - For homography objects: uses RANSAC-based generic PnP
        """
        state = self.targets[obj_name]
        image_points = corners_2d.reshape(-1, 1, 2).astype(np.float32)

        if state["method"] == "aruco":
            # Solve PnP with IPPE (more stable for planar squares)
            success, rvec, tvec = cv2.solvePnP(
                state["object_points_3d"], image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                return None, None
            return rvec, tvec
        else:
            # Generic PnP with RANSAC for homography case
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                state["object_points_3d"], image_points,
                self.camera_matrix, self.dist_coeffs,
                reprojectionError=8.0, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success or inliers is None or len(inliers) < 3:
                return None, None
            return rvec, tvec

    def rotation_to_euler(self, rvec):
        """
        Convert a rotation vector to Euler angles (roll, pitch, yaw) in degrees.
        Assumes standard XYZ convention.
        """
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        
        # Check for gimbal lock / singularity
        if sy > 1e-6:
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = 0
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    # ========== DRAWING ==========

    def draw_detection_box(self, frame, obj_name, bbox):
        """
        Draw YOLO detection bounding box for an object.
        Homography objects show different colors depending on calibration status.
        """
        x1, y1, x2, y2 = bbox
        state = self.targets[obj_name]
        
        # For homography objects: blue pre-calibration, green post-calibration
        if state["method"] == "homography":
            color = (0, 255, 0) if state.get("calibrated") else (255, 0, 0)
        else:
            # ArUco objects use green when detected
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{obj_name} ({state['method']})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_tracked_corners(self, frame, obj_name, corners):
        """
        Draw the 4 tracked corners of an object as a polyline.
        """
        corners_int = corners.astype(int)
        color = self.targets[obj_name]["axis_color"]
        cv2.polylines(frame, [corners_int], True, color, 3)
        return frame

    def draw_3d_axes(self, frame, obj_name, rvec, tvec, length=50):
        """
        Draw 3D coordinate axes (X,Y,Z) on the image for the object pose.
        X = red, Y = green, Z = blue; origin is filled with axis_color.
        """
        color = self.targets[obj_name]["axis_color"]
        
        # Define 3D axis endpoints (in mm)
        axis_points_3d = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
        axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        origin = tuple(axis_points_2d[0])
        
        # Draw X (red), Y (green), Z (blue) axes
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3, tipLength=0.3)
        # Draw origin as colored circle
        cv2.circle(frame, origin, 8, color, -1)
        
        return frame

    def draw_pose_info(self, frame, obj_name, y_offset_start, rvec_final, tvec_final):
        """
        Draw text overlay with pose information:
        - Object name
        - Position in mm
        - Roll/pitch/yaw in degrees
        Returns updated y-offset for subsequent objects.
        """
        roll, pitch, yaw = self.rotation_to_euler(rvec_final)
        text_color = self.targets[obj_name]["axis_color"]
        y_offset = y_offset_start
        
        cv2.putText(frame, f"--- {obj_name.upper()} ---", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        y_offset += 20
        
        cv2.putText(frame, f"Pos: [{tvec_final[0,0]:.0f}, {tvec_final[1,0]:.0f}, {tvec_final[2,0]:.0f}]mm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 18
        
        cv2.putText(frame, f"R/P/Y: [{roll:.1f}, {pitch:.1f}, {yaw:.1f}]deg",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 25
        
        return y_offset

    # ========== MAIN LOOP ==========

    def run(self):
        """
        Main processing loop:
        - Start camera
        - Grab frames
        - Run YOLO adaptively (search vs tracking)
        - Per-object: homography or ArUco pipeline
        - Estimate pose, smooth it, publish TF, draw overlays
        - Handle keyboard commands
        """
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("COMBINED POSE ESTIMATION SYSTEM + ROS2 TF")
        self.get_logger().info("5 Objects: 3 Homography + 2 ArUco")
        self.get_logger().info("="*60)
        self.get_logger().info("\nControls:")
        self.get_logger().info("  'r' - Reset homography calibrations")
        self.get_logger().info("  's' - Save frame")
        self.get_logger().info("  ESC - Exit")
        self.get_logger().info("="*60 + "\n")
        
        # Start camera pipeline; exit if startup fails
        if not self.start_camera():
            return
        
        frame_count = 0  # For FPS computation
        saved_count = 0  # Number of saved frames
        fps = 0
        last_time = time.time()
        
        try:
            # Main loop runs while ROS2 is active
            while rclpy.ok():
                frame = self.pull_frame()
                if frame is None:
                    # If no frame, still allow ESC to exit
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                
                frame_count += 1
                display_frame = frame.copy()
                
                # --- FPS calculation ---
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # --- Adaptive YOLO scheduling ---
                self.frame_idx += 1
                # If any homography object is calibrated, we can reduce YOLO rate
                any_calibrated = any(
                    state.get("calibrated", True) for state in self.targets.values()
                )
                mode_every_n = self.yolo_every_n_tracking if any_calibrated else self.yolo_every_n_search
                run_yolo = (self.frame_idx % mode_every_n == 0)
                
                # Track how many frames passed with no detections
                if not any(b is not None for b in self.last_bboxes.values()):
                    self.no_det_frames += 1
                else:
                    self.no_det_frames = 0
                
                # If lost track for many frames, force YOLO
                if self.no_det_frames > 10:
                    run_yolo = True
                    self.frame_idx = 0
                
                # Run YOLO or reuse previous detections
                if run_yolo:
                    bboxes, confidences = self.detect_objects(frame)
                    self.last_bboxes = bboxes
                    self.last_confidences = confidences
                else:
                    bboxes = self.last_bboxes
                    confidences = self.last_confidences
                
                # Text overlay y-position for pose info
                y_info_offset = 30
                # Count how many objects are currently tracked (for status display)
                tracked_count = 0
                
                # --- Process each detected object ---
                for obj_name, bbox in bboxes.items():
                    if bbox is None:
                        continue
                        
                    state = self.targets[obj_name]
                    # Draw detection bounding box
                    display_frame = self.draw_detection_box(display_frame, obj_name, bbox)
                    
                    # === HOMOGRAPHY PATH ===
                    if state["method"] == "homography":
                        # If not yet calibrated, handle calibration logic and UI hints
                        if not state.get("calibrated"):
                            conf = confidences.get(obj_name, 0.0)
                            
                            buffer_size = len(state.get("calib_buffer", []))
                            required = state.get("calib_buffer_size", 3)
                            
                            # Show calibration progress if buffer has frames
                            if buffer_size > 0:
                                cv2.putText(display_frame, f"{obj_name}: CALIBRATING {buffer_size}/{required}", 
                                          (10, y_info_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                y_info_offset += 30
                                cv2.putText(display_frame, "HOLD OBJECT STEADY!", 
                                          (10, y_info_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                y_info_offset += 35
                            # Show "ready" hint if conditions good for auto-calibration
                            elif self.should_auto_calibrate(obj_name, bbox, conf, frame.shape):
                                cv2.putText(display_frame, f"{obj_name}: Ready to calibrate", 
                                          (10, y_info_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                y_info_offset += 30
                            else:
                                # Explain why calibration cannot start yet
                                reason = ""
                                if conf < 0.75:
                                    reason = f"Low conf: {conf:.2f} < 0.75"
                                else:
                                    x1, y1, x2, y2 = bbox
                                    area = (x2 - x1) * (y2 - y1)
                                    frame_area = frame.shape[0] * frame.shape[1]
                                    area_ratio = area / frame_area
                                    if area_ratio < 0.08:
                                        reason = f"Too small: {area_ratio*100:.1f}% < 8%"
                                    else:
                                        reason = "Bad position/aspect"
                                
                                cv2.putText(display_frame, f"{obj_name}: NOT READY - {reason}", 
                                          (10, y_info_offset),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                                y_info_offset += 25
                            
                            # Try to accumulate frames toward calibration if conditions are met
                            if self.should_auto_calibrate(obj_name, bbox, conf, frame.shape):
                                self.accumulate_and_maybe_calibrate(frame, obj_name, bbox)
                            # Skip pose estimation until calibration is ready
                            continue
                        
                        # Once calibrated, track with homography to get 2D corners
                        corners = self.track_object_homography(frame, obj_name, bbox)
                        if corners is None:
                            continue
                            
                        # Estimate pose from 2D corners
                        rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                        if rvec_raw is None:
                            continue
                    
                    # === ARUCO PATH ===
                    elif state["method"] == "aruco":
                        # Detect markers in expanded ROI
                        aruco_corners, aruco_ids = self.detect_aruco_in_roi(frame, bbox)
                        if aruco_corners is None or aruco_ids is None:
                            continue
                        
                        target_id = state["aruco_id"]
                        # Skip if our target ID is not in detected IDs
                        if target_id not in aruco_ids:
                            continue
                        
                        # Get index of our specific marker
                        idx = np.where(aruco_ids == target_id)[0][0]
                        corners = aruco_corners[idx][0]
                        
                        # Estimate pose from marker corners
                        rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                        if rvec_raw is None:
                            continue
                    
                    # === COMMON: EMA SMOOTHING ===
                    alpha = state["ema_alpha"]
                    # Initialize smoothing state on first detection
                    if state["rvec_smooth"] is None:
                        state["rvec_smooth"] = rvec_raw.copy()
                        state["tvec_smooth"] = tvec_raw.copy()
                    else:
                        # Exponential moving average for smoother pose
                        state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                        state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                    
                    rvec_final = state["rvec_smooth"]
                    tvec_final = state["tvec_smooth"]
                    
                    # === PUBLISH TF ===
                    self.publish_tf(obj_name, rvec_final, tvec_final)
                    
                    # === DRAW ===
                    display_frame = self.draw_tracked_corners(display_frame, obj_name, corners)
                    display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final)
                    y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, rvec_final, tvec_final)
                    tracked_count += 1
                
                # --- UI overlays ---
                # Show FPS in top-right
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show tracking summary at bottom
                if tracked_count > 0:
                    cv2.putText(display_frame, f"TRACKING {tracked_count}/5 | PUBLISHING TF", 
                                (10, display_frame.shape[0]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the visualization window
                cv2.imshow('Combined Pose Estimation + ROS2', display_frame)
                
                # --- Keyboard handling ---
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:
                    # ESC key: exit cleanly
                    self.get_logger().info("\nExiting...")
                    break
                
                elif key == ord('r'):
                    # 'r': reset homography calibration for all homography objects
                    for obj_name, state in self.targets.items():
                        if state["method"] == "homography":
                            state["calibrated"] = False
                            state["ref_image"] = None
                            state["rvec_smooth"] = None
                            state["tvec_smooth"] = None
                            state["calib_buffer"].clear()
                    self.get_logger().info("✓ Homography calibrations reset")
                
                elif key == ord('s'):
                    # 's': save current display frame as JPEG
                    filename = f"combined_result_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    self.get_logger().info(f"✓ Saved: {filename}")
                
                # Spin ROS2 callbacks without blocking (timers, subscriptions, etc.)
                rclpy.spin_once(self, timeout_sec=0)
        
        finally:
            # Cleanup GStreamer pipeline and OpenCV windows on exit
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()


def main():
    """
    Entry point:
    - Initialize ROS2
    - Create CombinedPoseEstimatorROS2 node
    - Print configured objects
    - Run main loop
    """
    # Initialize ROS2 context
    rclpy.init()
    
    print("\n" + "="*60)
    print("COMBINED POSE ESTIMATION SETUP + ROS2")
    print("="*60)
    print("✓ Camera calibration from camera_params.py")
    
    # Path to YOLO model weights and camera index (not used in current pipeline)
    yolo_model_path = 'runs/detect/yolov8n_detect_V1/weights/best.pt'
    camera_index = 1
    
    # Create estimator node
    estimator = CombinedPoseEstimatorROS2(yolo_model_path, camera_index)
    
    # Print summary of object configurations to console
    print("\n--- Object Configurations ---")
    for obj_name, state in estimator.targets.items():
        method = state["method"]
        if method == "homography":
            print(f"→ {obj_name}: {state['width_mm']}x{state['height_mm']}mm ({method})")
        else:
            print(f"→ {obj_name}: ArUco ID {state['aruco_id']}, {state['marker_size_mm']}mm ({method})")
    print("="*60 + "\n")

    try:
        # Run main pose estimation loop
        estimator.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        # Always destroy node and shutdown ROS2 cleanly
        estimator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
