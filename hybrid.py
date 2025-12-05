#!/usr/bin/env python3
"""
HYBRID PNP Pose Estimation System with ROS2 TF Broadcasting
Combines Script 1's pipeline with Script 2's superior pose accuracy

KEY IMPROVEMENTS:
- Uses many feature points for PnP (like Script 2) instead of just 4 corners
- Direct 3D-2D correspondence without homography decomposition
- Reprojection error validation
- Translation jump guard
- Inlier refinement

Tracks all 5 objects:
- Hybrid PnP: card_game, circuit_board, notebook (50+ points each)
- ArUco: phone, estop (INDEPENDENT DETECTION)
- Publishes TF transforms to ROS2
- Uses Raspberry Pi CSI Camera via GStreamer
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
import time

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations

from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ==============================================================================
# STABILITY THRESHOLDS (from Script 2)
# ==============================================================================
REPROJ_ERROR_THRESH = 3.0      # pixels - reject frames with high reprojection error
TRANSLATION_JUMP_THRESH = 0.05 # meters (5 cm) - reject sudden large jumps
MIN_PNP_INLIERS = 6            # minimum inliers required for valid pose

# ==============================================================================
# ARUCO CONFIG
# ==============================================================================
ARUCO_DICT = cv2.aruco.DICT_4X4_50

ARUCO_IDS = {
    "phone": 0,
    "estop": 1,
}

ARUCO_MARKER_SIZES = {
    "phone": 37.0,   # mm
    "estop": 20.0,  # mm
}

# ==============================================================================
# OBJECT CONFIGS
# ==============================================================================
OBJECT_CONFIGS = {
    # Hybrid PnP objects (formerly "homography")
    "card_game": {
        "label_substring": "card_game",
        "width_mm": 93.0,
        "height_mm": 115.0,
        "axis_color": (0, 255, 0),       # Green
        "method": "hybrid_pnp",
        "min_matches": 8,                 # Reduced - need fewer with better method
        "min_inlier_ratio": 0.25,
        "ema_alpha": 0.2,
    },
    "circuit_board": {
        "label_substring": "circuit_board",
        "width_mm": 210.0,
        "height_mm": 210.0,
        "axis_color": (255, 0, 0),       # Blue
        "method": "hybrid_pnp",
        "min_matches": 10,
        "min_inlier_ratio": 0.4,
        "ema_alpha": 0.2,
    },
    "notebook": {
        "label_substring": "notebook",
        "width_mm": 147.0,
        "height_mm": 209.0,
        "axis_color": (0, 0, 255),       # Red
        "method": "hybrid_pnp",
        "min_matches": 10,
        "min_inlier_ratio": 0.4,
        "ema_alpha": 0.2,
    },
    # ArUco objects
    "phone": {
        "label_substring": "phone",
        "axis_color": (255, 255, 0),     # Cyan
        "method": "aruco",
        "aruco_id": ARUCO_IDS["phone"],
        "marker_size_mm": ARUCO_MARKER_SIZES["phone"],
        "ema_alpha": 0.3,
    },
    "estop": {
        "label_substring": "estop",
        "axis_color": (255, 0, 255),     # Magenta
        "method": "aruco",
        "aruco_id": ARUCO_IDS["estop"],
        "marker_size_mm": ARUCO_MARKER_SIZES["estop"],
        "ema_alpha": 0.3,
    },
}
# ==============================================================================


class HybridPnPPoseEstimatorROS2(Node):
    def __init__(self, yolo_model_path, camera_index=1):
        # Initialize ROS2 node
        super().__init__('hybrid_pnp_pose_estimator')
        
        # TF Broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.camera_frame = "camera_link_G4"
        
        self.get_logger().info("="*60)
        self.get_logger().info("HYBRID PNP ROS2 Pose Estimator Node Started")
        self.get_logger().info("Using Script 2's superior pose accuracy method")
        self.get_logger().info("="*60)
        
        # GStreamer pipeline for RPi CSI Camera
        self.pipeline = None
        self.sink = None
        self.camera_index = camera_index

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.fuse()
        
        self.yolo_imgsz = 384
        self.class_names = self.yolo_model.names
        self.get_logger().info(f"Model loaded with classes: {list(self.class_names.values())}")

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_aruco_api = True
            self.get_logger().info(f"✓ ArUco detector initialized (new API)")
        except AttributeError:
            self.use_new_aruco_api = False
            self.get_logger().info(f"✓ ArUco detector initialized (legacy API)")

        # Frame skipping
        self.frame_idx = 0
        self.yolo_every_n_tracking = 3
        self.yolo_every_n_search = 1
        self.no_det_frames = 0
        
        self.last_bboxes = {name: None for name in OBJECT_CONFIGS}
        self.last_confidences = {name: 0.0 for name in OBJECT_CONFIGS}

        # Build per-object state
        self.targets = {}

        for obj_name, cfg in OBJECT_CONFIGS.items():
            label_sub = cfg["label_substring"].lower()
            class_id = next(
                (cid for cid, cname in self.class_names.items()
                 if label_sub in cname.lower()),
                None,
            )
            if class_id is None:
                self.get_logger().warn(f"'{obj_name}' class not found in model!")
            else:
                self.get_logger().info(f"✓ {obj_name.capitalize()} class found: ID={class_id} ({cfg['method']})")

            state = {
                "class_id": class_id,
                "method": cfg["method"],
                "axis_color": cfg["axis_color"],
                "ema_alpha": cfg.get("ema_alpha", 0.2),
                "rvec_smooth": None,
                "tvec_smooth": None,
            }

            if cfg["method"] == "hybrid_pnp":
                width_mm = cfg["width_mm"]
                height_mm = cfg["height_mm"]

                state.update({
                    "width_mm": width_mm,
                    "height_mm": height_mm,
                    "min_matches": cfg.get("min_matches", 10),
                    "min_inlier_ratio": cfg.get("min_inlier_ratio", 0.4),
                    "ref_image": None,
                    "ref_keypoints": None,
                    "ref_descriptors": None,
                    "ref_keypoints_3d": None,  # NEW: 3D coordinates for each keypoint
                    "calibrated": False,
                    "calib_buffer": [],
                    "calib_buffer_size": 3,
                })

            elif cfg["method"] == "aruco":
                marker_size = cfg["marker_size_mm"]
                half_size = marker_size / 2.0

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

        # Camera parameters
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DIST_COEFFS
        self.get_logger().info("✓ Loaded camera parameters from camera_params.py")

        # ORB for hybrid PnP objects
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.get_logger().info(f"✓ ORB initialized (2000 features)")

    def publish_tf(self, obj_name, rvec, tvec):
        """Publish TF transform for an object"""
        t = TransformStamped()
        
        # Header
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = f"{obj_name}_frame"
        
        # Translation (convert mm to meters)
        t.transform.translation.x = float(tvec[0, 0]) / 1000.0
        t.transform.translation.y = float(tvec[1, 0]) / 1000.0
        t.transform.translation.z = float(tvec[2, 0]) / 1000.0
        
        # Rotation (convert rotation vector to quaternion)
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        quat = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([rotation_matrix, [[0], [0], [0]]]), [0, 0, 0, 1]])
        )
        
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def start_camera(self):
        """Start Raspberry Pi CSI Camera via GStreamer"""
        self.get_logger().info(f"Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
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
            self.get_logger().info(f"✓ GStreamer pipeline STARTED: 640x480")
            return True
        except Exception as e:
            self.get_logger().error(f"Cannot start GStreamer pipeline: {e}")
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

    # ========== YOLO DETECTION ==========
    
    def run_yolo_detection(self, frame):
        """Run YOLO detection on frame"""
        results = self.yolo_model(frame, imgsz=self.yolo_imgsz, verbose=False)
        
        bboxes = {}
        confidences = {}
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                for obj_name, state in self.targets.items():
                    if state["class_id"] == cls_id:
                        if obj_name not in bboxes or conf > confidences[obj_name]:
                            bboxes[obj_name] = [x1, y1, x2, y2]
                            confidences[obj_name] = conf
        
        return bboxes, confidences

    # ========== CALIBRATION (ENHANCED WITH 3D MAPPING) ==========
    
    def calibrate_reference(self, roi_or_frame, obj_name, bbox):
        """
        Enhanced calibration that stores 3D coordinates for each keypoint
        This is the key improvement from Script 2!
        """
        state = self.targets[obj_name]
        
        self.get_logger().info(f"\nCalibrating {obj_name.upper()}...")
        
        # Extract ROI
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = roi_or_frame[y1:y2, x1:x2]
        else:
            roi = roi_or_frame
        
        # Detect features
        state["ref_image"] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        state["ref_keypoints"], state["ref_descriptors"] = self.orb.detectAndCompute(
            state["ref_image"], None
        )
        
        if state["ref_descriptors"] is None or len(state["ref_keypoints"]) < 10:
            self.get_logger().warn("Not enough features! Retrying...")
            state["calibrated"] = False
            return False
        
        # ====== NEW: MAP EACH KEYPOINT TO 3D COORDINATES (Script 2 style) ======
        h_roi, w_roi = state["ref_image"].shape
        width_mm = state["width_mm"]
        height_mm = state["height_mm"]
        
        # Scale factors: pixels → millimeters
        scale_x = width_mm / w_roi
        scale_y = height_mm / h_roi
        
        # For each keypoint, compute 3D position on object plane (Z=0)
        state["ref_keypoints_3d"] = []
        for kp in state["ref_keypoints"]:
            u, v = kp.pt  # Pixel coordinates in reference image
            
            # Map to 3D object coordinates (centered at origin)
            # Reference image spans [0, width_mm] x [0, height_mm]
            # Center at origin: [-width_mm/2, width_mm/2] x [-height_mm/2, height_mm/2]
            X = (u * scale_x) - (width_mm / 2.0)
            Y = -((v * scale_y) - (height_mm / 2.0))  # Flip Y (image coords go down)
            Z = 0.0
            
            state["ref_keypoints_3d"].append([X, Y, Z])
        
        state["ref_keypoints_3d"] = np.array(state["ref_keypoints_3d"], dtype=np.float32)
        # ========================================================================
        
        state["rvec_smooth"] = None
        state["tvec_smooth"] = None
        state["calibrated"] = True
        
        self.get_logger().info(f"✓ Detected {len(state['ref_keypoints'])} features")
        self.get_logger().info(f"✓ Mapped {len(state['ref_keypoints_3d'])} keypoints to 3D")
        self.get_logger().info(f"✓ CALIBRATION COMPLETE for {obj_name.upper()}!")
        self.get_logger().info('='*60 + '\n')
        return True

    def should_auto_calibrate(self, obj_name, bbox, confidence, frame_shape):
        """Check if suitable for calibration - STRICT CONDITIONS"""
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
        
        margin = 30
        if x1 < margin or y1 < margin or x2 > (W - margin) or y2 > (H - margin):
            return False
        
        aspect = max(w, h) / min(w, h)
        if aspect > 3.0:
            return False

        return True

    def accumulate_and_maybe_calibrate(self, frame, obj_name, bbox):
        """Buffer and calibrate - requires STABILITY"""
        state = self.targets[obj_name]
        x1, y1, x2, y2 = bbox
        
        if len(state["calib_buffer"]) > 0:
            last_bbox = state["calib_buffer"][-1]["bbox"]
            dx = abs((x1 + x2) / 2 - (last_bbox[0] + last_bbox[2]) / 2)
            dy = abs((y1 + y2) / 2 - (last_bbox[1] + last_bbox[3]) / 2)
            
            if dx > 20 or dy > 20:
                self.get_logger().info(f"Movement detected for {obj_name} - resetting calibration buffer")
                state["calib_buffer"].clear()
                return False
        
        roi = frame[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            return False

        state["calib_buffer"].append({
            "roi": roi, 
            "n_kp": len(keypoints),
            "bbox": [x1, y1, x2, y2]
        })

        if len(state["calib_buffer"]) < state["calib_buffer_size"]:
            self.get_logger().info(f"Calibration buffer for {obj_name}: {len(state['calib_buffer'])}/{state['calib_buffer_size']} "
                  f"(keypoints: {len(keypoints)}) - HOLD STEADY!")
            return False

        best_candidate = max(state["calib_buffer"], key=lambda c: c["n_kp"])
        roi_best = best_candidate["roi"]
        h_best, w_best = roi_best.shape[:2]
        
        success = self.calibrate_reference(roi_best, obj_name, [0, 0, w_best, h_best])
        state["calib_buffer"].clear()
        return success

    # ========== HYBRID PNP TRACKING (THE KEY IMPROVEMENT!) ==========
    
    def track_object_hybrid_pnp(self, frame, obj_name, bbox):
        """
        HYBRID PNP TRACKING - Direct 3D-2D correspondence (Script 2 style)
        
        This is the superior method that uses many feature points for PnP
        instead of just 4 corners from homography decomposition.
        
        Returns: dict with pose and validation info, or None if tracking failed
        """
        state = self.targets[obj_name]
        
        if not state.get("calibrated") or state["ref_descriptors"] is None:
            return None
        
        min_matches = state["min_matches"]
        
        # Extract ROI with padding
        x1_raw, y1_raw, x2_raw, y2_raw = bbox
        pad = 10
        h, w, _ = frame.shape
        x1 = max(0, x1_raw - pad)
        y1 = max(0, y1_raw - pad)
        x2 = min(w - 1, x2_raw + pad)
        y2 = min(h - 1, y2_raw + pad)
        
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect features in current frame
        keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)
        
        if descriptors is None or len(keypoints) < 10:
            return None
        
        # Match features
        matches = self.bf_matcher.knnMatch(state["ref_descriptors"], descriptors, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
        
        if len(good_matches) < min_matches:
            return None
        
        # ====== KEY DIFFERENCE: Build 3D-2D correspondences from matches ======
        # This is what Script 2 does - direct 3D-2D mapping!
        obj_pts_3d = []
        img_pts_2d = []
        
        for match in good_matches:
            # Get 3D point from reference (stored during calibration)
            pt_3d = state["ref_keypoints_3d"][match.queryIdx]
            
            # Get 2D point from current frame
            pt_2d = keypoints[match.trainIdx].pt
            
            # Adjust 2D point to frame coordinates (not ROI)
            pt_2d_frame = (pt_2d[0] + x1, pt_2d[1] + y1)
            
            obj_pts_3d.append(pt_3d)
            img_pts_2d.append(pt_2d_frame)
        
        obj_pts_3d = np.array(obj_pts_3d, dtype=np.float32)
        img_pts_2d = np.array(img_pts_2d, dtype=np.float32)
        # =====================================================================
        
        # Direct PnP with MANY points (not just 4 corners!)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts_3d, 
            img_pts_2d,
            self.camera_matrix, 
            self.dist_coeffs,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            return None
        
        # Extract inliers
        inliers = inliers.ravel()
        obj_pts_inliers = obj_pts_3d[inliers]
        img_pts_inliers = img_pts_2d[inliers]
        
        # ====== INLIER REFINEMENT (Script 2 technique) ======
        # Compute reprojection errors for inliers
        proj_pts, _ = cv2.projectPoints(
            obj_pts_inliers, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        proj_pts = proj_pts.reshape(-1, 2)
        reproj_errors = np.linalg.norm(proj_pts - img_pts_inliers, axis=1)
        
        # Keep only low-error inliers for refinement
        low_error_idx = np.where(reproj_errors < REPROJ_ERROR_THRESH)[0]
        
        if len(low_error_idx) >= MIN_PNP_INLIERS:
            obj_pts_refined = obj_pts_inliers[low_error_idx]
            img_pts_refined = img_pts_inliers[low_error_idx]
            
            # Re-solve PnP with refined inliers
            try:
                success_ref, rvec_ref, tvec_ref = cv2.solvePnP(
                    obj_pts_refined, img_pts_refined,
                    self.camera_matrix, self.dist_coeffs,
                    rvec, tvec, True,  # Use previous as initial guess
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success_ref:
                    rvec = rvec_ref
                    tvec = tvec_ref
                    obj_pts_inliers = obj_pts_refined
                    img_pts_inliers = img_pts_refined
                    
                    # Recompute errors with refined pose
                    proj_pts, _ = cv2.projectPoints(
                        obj_pts_inliers, rvec, tvec,
                        self.camera_matrix, self.dist_coeffs
                    )
                    proj_pts = proj_pts.reshape(-1, 2)
                    reproj_errors = np.linalg.norm(proj_pts - img_pts_inliers, axis=1)
            except cv2.error:
                pass  # Keep original if refinement fails
        # ====================================================
        
        mean_reproj_error = np.mean(reproj_errors)
        inlier_ratio = len(inliers) / len(obj_pts_3d)
        
        # Return pose with validation metrics
        return {
            "rvec": rvec,
            "tvec": tvec,
            "inlier_count": len(obj_pts_inliers),
            "inlier_ratio": inlier_ratio,
            "mean_reproj_error": mean_reproj_error,
            "obj_pts": obj_pts_inliers,
            "img_pts": img_pts_inliers,
        }

    # ========== ARUCO METHODS ==========
    
    def detect_aruco_independent(self, frame):
        """Detect all ArUco markers in entire frame (independent of YOLO)"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_aruco_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray_frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray_frame, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None or len(ids) == 0:
            return {}
        
        # Build dictionary of detected ArUco objects
        detections = {}
        for obj_name, state in self.targets.items():
            if state["method"] != "aruco":
                continue
            target_id = state["aruco_id"]
            if target_id in ids.flatten():
                idx = np.where(ids.flatten() == target_id)[0][0]
                detections[obj_name] = corners[idx][0]
        
        return detections

    def estimate_pose_aruco(self, obj_name, corners_2d):
        """Estimate pose for ArUco marker"""
        state = self.targets[obj_name]
        image_points = corners_2d.reshape(-1, 1, 2).astype(np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            state["object_points_3d"], image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if not success:
            return None, None
        
        return rvec, tvec

    # ========== VALIDATION (Script 2 techniques) ==========
    
    def validate_pose_reprojection(self, obj_name, pose_result):
        """
        Validate pose using reprojection error (Script 2 technique)
        Returns: (is_valid, error_message)
        """
        mean_error = pose_result["mean_reproj_error"]
        
        if mean_error > REPROJ_ERROR_THRESH:
            return False, f"High reproj error: {mean_error:.2f}px"
        
        return True, ""
    
    def validate_translation_jump(self, obj_name, tvec_new):
        """
        Detect sudden large movements (Script 2 technique)
        Returns: (is_valid, error_message)
        """
        state = self.targets[obj_name]
        
        if state["tvec_smooth"] is None:
            return True, ""  # First frame, accept
        
        jump = np.linalg.norm(tvec_new - state["tvec_smooth"])
        
        if jump > TRANSLATION_JUMP_THRESH:
            return False, f"Large jump: {jump*1000:.1f}mm"
        
        return True, ""

    # ========== VISUALIZATION ==========
    
    def draw_detection_box(self, frame, obj_name, bbox, confidence):
        """Draw bounding box for detected object"""
        if confidence < 0.6:
            return frame
        
        x1, y1, x2, y2 = bbox
        state = self.targets[obj_name]
        color = state["axis_color"]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{obj_name}: {confidence:.2f}"
        (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - h_text - 4), (x1 + w_text, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def draw_matched_points(self, frame, img_pts, color):
        """Draw matched feature points"""
        for pt in img_pts:
            cv2.circle(frame, tuple(pt.astype(int)), 3, color, -1)
        return frame
    
    def draw_3d_axes(self, frame, obj_name, rvec, tvec):
        """Draw 3D coordinate axes"""
        state = self.targets[obj_name]
        
        if state["method"] == "hybrid_pnp":
            axis_len = min(state["width_mm"], state["height_mm"]) * 0.5
        else:
            axis_len = state["marker_size_mm"] * 0.5
        
        axis_3d = np.float32([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len]
        ])
        
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec,
                                      self.camera_matrix, self.dist_coeffs)
        imgpts = imgpts.reshape(-1, 2)
        
        origin = tuple(imgpts[0].astype(int))
        x_end = tuple(imgpts[1].astype(int))
        y_end = tuple(imgpts[2].astype(int))
        z_end = tuple(imgpts[3].astype(int))
        
        cv2.line(frame, origin, x_end, (0, 0, 255), 3)  # X: Red
        cv2.line(frame, origin, y_end, (0, 255, 0), 3)  # Y: Green
        cv2.line(frame, origin, z_end, (255, 0, 0), 3)  # Z: Blue
        
        return frame
    
    def draw_pose_info(self, frame, obj_name, y_offset, rvec, tvec, extra_info=""):
        """Draw pose information text"""
        t = tvec.ravel()
        
        cv2.putText(frame, f"{obj_name}: X={t[0]/1000:.3f}m Y={t[1]/1000:.3f}m Z={t[2]/1000:.3f}m",
                   (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
        
        if extra_info:
            cv2.putText(frame, f"  {extra_info}",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
        
        return y_offset

    # ========== MAIN LOOP ==========
    
    def run(self):
        """Main processing loop"""
        if not self.start_camera():
            return
        
        fps = 0
        prev_time = time.time()
        saved_count = 0
        frame_count = 0
        
        try:
            while True:
                frame = self.pull_frame()
                if frame is None:
                    continue
                
                # FPS calculation
                frame_count += 1
                curr_time = time.time()
                if curr_time - prev_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    prev_time = curr_time
                
                display_frame = frame.copy()
                
                # Frame skipping logic
                self.frame_idx += 1
                
                any_tracking = any(
                    state.get("calibrated", False) 
                    for state in self.targets.values() 
                    if state["method"] == "hybrid_pnp"
                )
                
                yolo_interval = self.yolo_every_n_tracking if any_tracking else self.yolo_every_n_search
                run_yolo = (self.frame_idx % yolo_interval == 0)
                
                # Run YOLO detection
                if run_yolo:
                    bboxes, confidences = self.run_yolo_detection(frame)
                    
                    for obj_name in self.targets:
                        if self.targets[obj_name]["method"] == "hybrid_pnp":
                            if obj_name in bboxes:
                                self.last_bboxes[obj_name] = bboxes[obj_name]
                                self.last_confidences[obj_name] = confidences[obj_name]
                else:
                    bboxes = self.last_bboxes
                    confidences = self.last_confidences
                
                # Detect ArUco markers (independent)
                aruco_detections = self.detect_aruco_independent(frame)
                
                # Track objects
                tracked_count = 0
                y_info_offset = 30
                
                # Process hybrid PnP objects
                for obj_name, state in self.targets.items():
                    if state["method"] != "hybrid_pnp":
                        continue
                    
                    bbox = bboxes.get(obj_name)
                    
                    if bbox is None:
                        continue
                    
                    # Draw bbox
                    conf = confidences.get(obj_name, 0.0)
                    display_frame = self.draw_detection_box(display_frame, obj_name, bbox, conf)
                    
                    # === CALIBRATION PHASE ===
                    if not state.get("calibrated"):
                        buffer_size = len(state.get("calib_buffer", []))
                        required = state.get("calib_buffer_size", 3)
                        
                        if buffer_size > 0:
                            cv2.putText(display_frame, f"{obj_name}: CALIBRATING {buffer_size}/{required}", 
                                      (10, y_info_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            y_info_offset += 30
                            cv2.putText(display_frame, "HOLD OBJECT STEADY!", 
                                      (10, y_info_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            y_info_offset += 35
                        elif self.should_auto_calibrate(obj_name, bbox, conf, frame.shape):
                            cv2.putText(display_frame, f"{obj_name}: Ready to calibrate", 
                                      (10, y_info_offset),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y_info_offset += 30
                        else:
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
                        
                        if self.should_auto_calibrate(obj_name, bbox, conf, frame.shape):
                            self.accumulate_and_maybe_calibrate(frame, obj_name, bbox)
                        continue
                    
                    # === TRACKING PHASE (HYBRID PNP!) ===
                    pose_result = self.track_object_hybrid_pnp(frame, obj_name, bbox)
                    
                    if pose_result is None:
                        continue
                    
                    rvec_raw = pose_result["rvec"]
                    tvec_raw = pose_result["tvec"]
                    
                    # === VALIDATION (Script 2 techniques) ===
                    valid_reproj, reproj_msg = self.validate_pose_reprojection(obj_name, pose_result)
                    valid_jump, jump_msg = self.validate_translation_jump(obj_name, tvec_raw)
                    
                    # Reject frame if validation fails
                    if not valid_reproj:
                        if state["rvec_smooth"] is not None:
                            self.get_logger().warn(f"{obj_name}: {reproj_msg}, using previous pose")
                            rvec_raw = state["rvec_smooth"]
                            tvec_raw = state["tvec_smooth"]
                        else:
                            continue
                    
                    if not valid_jump:
                        if state["tvec_smooth"] is not None:
                            self.get_logger().warn(f"{obj_name}: {jump_msg}, using previous pose")
                            rvec_raw = state["rvec_smooth"]
                            tvec_raw = state["tvec_smooth"]
                    
                    # === EMA SMOOTHING ===
                    alpha = state["ema_alpha"]
                    if state["rvec_smooth"] is None:
                        state["rvec_smooth"] = rvec_raw.copy()
                        state["tvec_smooth"] = tvec_raw.copy()
                    else:
                        state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                        state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                    
                    rvec_final = state["rvec_smooth"]
                    tvec_final = state["tvec_smooth"]
                    
                    # === PUBLISH TF ===
                    self.publish_tf(obj_name, rvec_final, tvec_final)
                    
                    # === DRAW ===
                    display_frame = self.draw_matched_points(
                        display_frame, 
                        pose_result["img_pts"], 
                        state["axis_color"]
                    )
                    display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final)
                    
                    extra_info = f"{pose_result['inlier_count']} pts, err={pose_result['mean_reproj_error']:.2f}px"
                    y_info_offset = self.draw_pose_info(
                        display_frame, obj_name, y_info_offset, 
                        rvec_final, tvec_final, extra_info
                    )
                    
                    tracked_count += 1
                
                # Process ArUco objects
                for obj_name, corners in aruco_detections.items():
                    state = self.targets[obj_name]
                    
                    rvec_raw, tvec_raw = self.estimate_pose_aruco(obj_name, corners)
                    if rvec_raw is None:
                        continue
                    
                    # === EMA SMOOTHING ===
                    alpha = state["ema_alpha"]
                    if state["rvec_smooth"] is None:
                        state["rvec_smooth"] = rvec_raw.copy()
                        state["tvec_smooth"] = tvec_raw.copy()
                    else:
                        state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                        state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                    
                    rvec_final = state["rvec_smooth"]
                    tvec_final = state["tvec_smooth"]
                    
                    # === PUBLISH TF ===
                    self.publish_tf(obj_name, rvec_final, tvec_final)
                    
                    # === DRAW ===
                    # Draw ArUco corners
                    for i in range(4):
                        pt1 = tuple(corners[i].astype(int))
                        pt2 = tuple(corners[(i+1)%4].astype(int))
                        cv2.line(display_frame, pt1, pt2, state["axis_color"], 2)
                    
                    display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final)
                    y_info_offset = self.draw_pose_info(
                        display_frame, obj_name, y_info_offset, 
                        rvec_final, tvec_final
                    )
                    
                    tracked_count += 1
                
                # UI
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if tracked_count > 0:
                    cv2.putText(display_frame, f"TRACKING {tracked_count}/5 | HYBRID PNP + ARUCO", 
                                (10, display_frame.shape[0]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Hybrid PnP Pose Estimation + ROS2', display_frame)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    self.get_logger().info("\nExiting...")
                    break
                
                elif key == ord('r'):  # Reset calibrations
                    for obj_name, state in self.targets.items():
                        if state["method"] == "hybrid_pnp":
                            state["calibrated"] = False
                            state["ref_image"] = None
                            state["ref_keypoints_3d"] = None
                            state["rvec_smooth"] = None
                            state["tvec_smooth"] = None
                            state["calib_buffer"].clear()
                    self.get_logger().info("✓ All calibrations reset")
                
                elif key == ord('s'):  # Save screenshot
                    filename = f"hybrid_pnp_result_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    self.get_logger().info(f"✓ Saved: {filename}")
                
                # Spin ROS2 callbacks
                rclpy.spin_once(self, timeout_sec=0)
        
        finally:
            # Cleanup
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()


def main():
    # Initialize ROS2
    rclpy.init()
    
    print("\n" + "="*60)
    print("HYBRID PNP POSE ESTIMATION + ROS2")
    print("Combining Script 1's pipeline with Script 2's accuracy")
    print("="*60)
    print("✓ Using many feature points for PnP (not just 4 corners)")
    print("✓ Direct 3D-2D correspondence (no homography ambiguity)")
    print("✓ Reprojection error validation")
    print("✓ Translation jump guard")
    print("✓ Inlier refinement")
    print("="*60)
    
    yolo_model_path = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
    camera_index = 1
    
    estimator = HybridPnPPoseEstimatorROS2(yolo_model_path, camera_index)
    
    print("\n--- Object Configurations ---")
    for obj_name, state in estimator.targets.items():
        method = state["method"]
        if method == "hybrid_pnp":
            print(f"→ {obj_name}: {state['width_mm']}x{state['height_mm']}mm ({method} - MANY POINTS)")
        else:
            print(f"→ {obj_name}: ArUco ID {state['aruco_id']}, {state['marker_size_mm']}mm ({method})")
    print("="*60 + "\n")

    try:
        estimator.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    finally:
        estimator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()