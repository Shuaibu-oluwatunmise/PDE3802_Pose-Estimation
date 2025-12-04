"""
ROS2 Pose Estimation - HYBRID APPROACH
- Runtime auto-calibration (5-frame buffer, pick best)
- ROI-based homography tracking (fast!)
- ArUco INDEPENDENT of YOLO (always scans)
- ROS2 TF publishing for all objects
- Adaptive frame skipping
- Raspberry Pi CSI Camera
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
from camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ROS2 Imports
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# ==============================================================================
# ARUCO CONFIG
# ==============================================================================
ARUCO_DICT = cv2.aruco.DICT_4X4_50

ARUCO_CONFIG = {
    "wallet": {
        "aruco_id": 0,
        "marker_size_mm": 50.0,
        "axis_color": (255, 255, 0),  # Cyan
        "ema_alpha": 0.3,
    },
    "headset": {
        "aruco_id": 1,
        "marker_size_mm": 32.0,
        "axis_color": (255, 0, 255),  # Magenta
        "ema_alpha": 0.3,
    },
}

# ==============================================================================
# HOMOGRAPHY OBJECT CONFIGS
# ==============================================================================
HOMOGRAPHY_CONFIGS = {
    "notebook": {
        "label_substring": "notebook",
        "width_mm": 170.0,
        "height_mm": 230.0,
        "axis_color": (0, 0, 255),  # Red
        "min_matches": 10,
        "min_inlier_ratio": 0.5,
        "ema_alpha": 0.2,
    },
    "game_box": {
        "label_substring": "game_box",
        "width_mm": 312.0,
        "height_mm": 146.0,
        "axis_color": (255, 0, 0),  # Blue
        "min_matches": 6,
        "min_inlier_ratio": 0.3,
        "ema_alpha": 0.1,
    },
    "repair_mat": {
        "label_substring": "repair_mat",
        "width_mm": 300.0,
        "height_mm": 200.0,
        "axis_color": (0, 255, 0),  # Green
        "min_matches": 8,
        "min_inlier_ratio": 0.35,
        "ema_alpha": 0.15,
    },
}


class HybridPoseEstimator(Node):
    def __init__(self, yolo_model_path):
        # ROS2 Node initialization
        super().__init__('hybrid_pose_estimation_node')
        
        # ROS2 TF broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.camera_frame = "camera_link"
        self._publish_static_camera_frame()
        self.get_logger().info("✓ ROS2 TF broadcasters initialized")
        
        # RPI CSI
        self.pipeline = None
        self.sink = None
        
        # Load YOLO
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.fuse()
        self.yolo_imgsz = 384
        self.class_names = self.yolo_model.names
        print(f"Model loaded with classes: {list(self.class_names.values())}")
        
        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_aruco_api = True
            print(f"✓ ArUco detector initialized (new API)")
        except AttributeError:
            self.use_new_aruco_api = False
            print(f"✓ ArUco detector initialized (legacy API)")
        
        # Adaptive frame skipping
        self.frame_idx = 0
        self.yolo_every_n_tracking = 6
        self.yolo_every_n_search = 1
        self.no_det_frames = 0
        
        self.last_bboxes = {name: None for name in HOMOGRAPHY_CONFIGS}
        self.last_confidences = {name: 0.0 for name in HOMOGRAPHY_CONFIGS}
        
        # Camera parameters
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DIST_COEFFS
        print("✓ Loaded camera parameters from camera_params.py")
        
        # ORB for homography
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=12)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print(f"✓ ORB optimized for Pi (1500 features)")
        
        # Build homography targets
        self.homography_targets = {}
        self._initialize_homography_targets()
        
        # Build ArUco targets
        self.aruco_targets = {}
        self._initialize_aruco_targets()
    
    def _publish_static_camera_frame(self):
        """Publish static camera_link frame"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = self.camera_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(t)
        self.get_logger().info("✓ Published static camera_link frame")
    
    def _initialize_homography_targets(self):
        """Initialize homography objects"""
        for obj_name, cfg in HOMOGRAPHY_CONFIGS.items():
            label_sub = cfg["label_substring"].lower()
            class_id = next(
                (cid for cid, cname in self.class_names.items()
                 if label_sub in cname.lower()),
                None,
            )
            
            if class_id is None:
                print(f"WARNING: '{obj_name}' class not found in model!")
                continue
            
            print(f"✓ {obj_name.capitalize()} class found: ID={class_id} (homography)")
            
            width_mm = cfg["width_mm"]
            height_mm = cfg["height_mm"]
            half_w = width_mm / 2.0
            half_h = height_mm / 2.0
            
            object_points_3d = np.array([
                [-half_w,  half_h, 0],
                [ half_w,  half_h, 0],
                [ half_w, -half_h, 0],
                [-half_w, -half_h, 0],
            ], dtype=np.float32)
            
            self.homography_targets[obj_name] = {
                "class_id": class_id,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "object_points_3d": object_points_3d,
                "axis_color": cfg["axis_color"],
                "min_matches": cfg.get("min_matches", 10),
                "min_inlier_ratio": cfg.get("min_inlier_ratio", 0.5),
                "ema_alpha": cfg.get("ema_alpha", 0.2),
                "ref_image": None,
                "ref_keypoints": None,
                "ref_descriptors": None,
                "calibrated": False,
                "rvec_smooth": None,
                "tvec_smooth": None,
                "calib_buffer": [],
                "calib_buffer_size": 5,  # 5 frames buffer
            }
    
    def _initialize_aruco_targets(self):
        """Initialize ArUco objects"""
        for obj_name, cfg in ARUCO_CONFIG.items():
            marker_size = cfg["marker_size_mm"]
            half_size = marker_size / 2.0
            
            object_points_3d = np.array([
                [-half_size,  half_size, 0],
                [ half_size,  half_size, 0],
                [ half_size, -half_size, 0],
                [-half_size, -half_size, 0],
            ], dtype=np.float32)
            
            self.aruco_targets[obj_name] = {
                "aruco_id": cfg["aruco_id"],
                "marker_size_mm": marker_size,
                "object_points_3d": object_points_3d,
                "axis_color": cfg["axis_color"],
                "ema_alpha": cfg.get("ema_alpha", 0.3),
                "rvec_smooth": None,
                "tvec_smooth": None,
            }
            
            print(f"✓ {obj_name.capitalize()}: ArUco ID {cfg['aruco_id']} (aruco)")
    
    def start_camera(self):
        """Start Raspberry Pi CSI Camera"""
        print(f"Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
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
            print(f"✓ GStreamer pipeline STARTED: 640x480")
            return True
        except Exception as e:
            print(f"ERROR: Cannot start GStreamer pipeline: {e}")
            return False
    
    def pull_frame(self, timeout_ns=10_000_000):
        """Grab frame from appsink"""
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
        """YOLO detection for homography objects"""
        results = self.yolo_model(frame, verbose=False, imgsz=self.yolo_imgsz, conf=0.3, iou=0.5)
        result = results[0]
        
        bboxes = {name: None for name in self.homography_targets}
        best_conf = {name: 0.0 for name in self.homography_targets}
        
        if result.boxes is None or len(result.boxes) == 0:
            return bboxes, best_conf
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            for name, state in self.homography_targets.items():
                if state["class_id"] is None:
                    continue
                if class_id == state["class_id"] and confidence > best_conf[name]:
                    bboxes[name] = [x1, y1, x2, y2]
                    best_conf[name] = confidence
        
        return bboxes, best_conf
    
    def detect_aruco_full_frame(self, frame):
        """Detect ALL ArUco markers in full frame - INDEPENDENT of YOLO"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_aruco_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None:
            return None, None
        
        return corners, ids.flatten()
    
    # Calibration methods (from Document 7)
    def calibrate_reference(self, frame, obj_name, bbox):
        """Calibrate reference for homography object"""
        state = self.homography_targets[obj_name]
        print("\n" + "="*60)
        print(f"CALIBRATING FROM BEST CANDIDATE: {obj_name.upper()}")
        print("="*60)
        
        x1, y1, x2, y2 = bbox
        
        if x1 == 0 and y1 == 0 and x2 == frame.shape[1] and y2 == frame.shape[0]:
            roi = frame
        else:
            roi = frame[y1:y2, x1:x2]
        
        state["ref_image"] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        state["ref_keypoints"], state["ref_descriptors"] = self.orb.detectAndCompute(state["ref_image"], None)
        
        if state["ref_descriptors"] is None or len(state["ref_keypoints"]) < 10:
            print("ERROR: Not enough features! Retrying.")
            state["calibrated"] = False
            return False
        
        state["rvec_smooth"] = None
        state["tvec_smooth"] = None
        
        print(f"✓ Detected {len(state['ref_keypoints'])} features")
        state["calibrated"] = True
        print("="*60)
        print(f"✓ CALIBRATION COMPLETE for {obj_name.upper()}!")
        print("="*60 + "\n")
        return True
    
    def should_auto_calibrate(self, obj_name, bbox, confidence, frame_shape):
        """Gate check for auto-calibration"""
        if confidence < 0.7:
            return False
        
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        H, W = frame_shape[:2]
        area = w * h
        frame_area = W * H
        
        if area < 0.02 * frame_area:
            return False
        
        return True
    
    def accumulate_and_maybe_calibrate(self, frame, obj_name, bbox):
        """Buffer 5 frames, calibrate from best"""
        state = self.homography_targets[obj_name]
        x1, y1, x2, y2 = bbox
        
        roi = frame[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 10:
            return False
        
        state["calib_buffer"].append({
            "roi": roi,
            "n_kp": len(keypoints),
        })
        
        if len(state["calib_buffer"]) < state["calib_buffer_size"]:
            print(f"Buffered candidate for {obj_name} ({len(state['calib_buffer'])}/{state['calib_buffer_size']})")
            return False
        
        print(f"Buffer full for {obj_name}. Selecting best frame...")
        
        best_candidate = max(state["calib_buffer"], key=lambda c: c["n_kp"])
        roi_best = best_candidate["roi"]
        
        h_best, w_best = roi_best.shape[:2]
        bbox_best = [0, 0, w_best, h_best]
        success = self.calibrate_reference(roi_best, obj_name, bbox_best)
        
        state["calib_buffer"].clear()
        
        return success
    
    def track_object_homography(self, frame, obj_name, bbox):
        """Track object in ROI using homography"""
        state = self.homography_targets[obj_name]
        
        if not state["calibrated"] or state["ref_descriptors"] is None:
            return None
        
        min_matches = state["min_matches"]
        min_inlier_ratio = state["min_inlier_ratio"]
        
        x1_raw, y1_raw, x2_raw, y2_raw = bbox
        
        pad = 10
        h, w, _ = frame.shape
        x1 = max(0, x1_raw - pad)
        y1 = max(0, y1_raw - pad)
        x2 = min(w - 1, x2_raw + pad)
        y2 = min(h - 1, y2_raw + pad)
        
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)
        
        if descriptors is None or len(keypoints) < 10:
            return None
        
        matches = self.bf_matcher.knnMatch(state["ref_descriptors"], descriptors, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < min_matches:
            return None
        
        src_pts = np.float32([state["ref_keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None
        
        inliers = mask.ravel().tolist()
        inlier_ratio = sum(inliers) / len(inliers)
        if inlier_ratio < min_inlier_ratio:
            return None
        
        h_ref, w_ref = state["ref_image"].shape
        ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)
        
        corners_roi = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)
        
        corners_frame = corners_roi.copy()
        corners_frame[:, 0] += x1
        corners_frame[:, 1] += y1
        
        return corners_frame
    
    def estimate_pose_pnp(self, object_points_3d, corners_2d):
        """Estimate 6DOF pose"""
        image_points = corners_2d.reshape(-1, 1, 2).astype(np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_3d,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            reprojectionError=8.0,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < 3:
            return None, None
        
        return rvec, tvec
    
    def rotation_to_euler(self, rvec):
        """Convert rotation vector to Euler angles"""
        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        
        if sy > 1e-6:
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = 0
        
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return [x, y, z, w]
    
    def publish_tf(self, obj_name, rvec, tvec):
        """Publish TF frame to ROS2"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = f"{obj_name}_frame"
        
        # Translation (mm to meters)
        t.transform.translation.x = float(tvec[0, 0]) / 1000.0
        t.transform.translation.y = float(tvec[1, 0]) / 1000.0
        t.transform.translation.z = float(tvec[2, 0]) / 1000.0
        
        # Rotation
        rmat, _ = cv2.Rodrigues(rvec)
        quat = self.rotation_matrix_to_quaternion(rmat)
        
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        
        self.tf_broadcaster.sendTransform(t)
    
    # Drawing methods
    def draw_detection_box(self, frame, obj_name, bbox, is_homography=True):
        """Draw YOLO/ArUco detection box"""
        x1, y1, x2, y2 = bbox
        
        if is_homography:
            state = self.homography_targets[obj_name]
            color = (0, 255, 0) if state["calibrated"] else (255, 0, 0)
        else:
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{obj_name}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
    
    def draw_tracked_corners(self, frame, obj_name, corners, is_homography=True):
        """Draw tracked corners"""
        corners_int = corners.astype(int)
        
        if is_homography:
            color = self.homography_targets[obj_name]["axis_color"]
        else:
            color = self.aruco_targets[obj_name]["axis_color"]
        
        cv2.polylines(frame, [corners_int], True, color, 3)
        return frame
    
    def draw_3d_axes(self, frame, obj_name, rvec, tvec, is_homography=True, length=50):
        """Draw 3D axes"""
        if is_homography:
            color = self.homography_targets[obj_name]["axis_color"]
        else:
            color = self.aruco_targets[obj_name]["axis_color"]
        
        axis_points_3d = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
        axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        origin = tuple(axis_points_2d[0])
        
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3, tipLength=0.3)
        cv2.circle(frame, origin, 8, color, -1)
        
        return frame
    
    def draw_pose_info(self, frame, obj_name, y_offset_start, rvec, tvec, is_homography=True):
        """Draw pose info"""
        roll, pitch, yaw = self.rotation_to_euler(rvec)
        distance_z = tvec[2, 0]
        
        if is_homography:
            text_color = self.homography_targets[obj_name]["axis_color"]
        else:
            text_color = self.aruco_targets[obj_name]["axis_color"]
        
        y_offset = y_offset_start
        
        cv2.putText(frame, f"--- {obj_name.upper()} ---", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        y_offset += 20
        
        cv2.putText(frame, f"Pos: [{tvec[0,0]:.0f}, {tvec[1,0]:.0f}, {distance_z:.0f}]mm", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 18
        
        cv2.putText(frame, f"R/P/Y: [{roll:.1f}, {pitch:.1f}, {yaw:.1f}]deg", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 25
        
        return y_offset
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("ROS2 HYBRID POSE ESTIMATION")
        print("="*60)
        print("\nFeatures:")
        print("  • Runtime auto-calibration (5-frame buffer)")
        print("  • ROI-based homography tracking")
        print("  • ArUco INDEPENDENT detection")
        print("  • ROS2 TF publishing")
        print("  • Adaptive frame skipping")
        print("\nControls:")
        print("  'r' - Reset calibrations")
        print("  's' - Save frame")
        print("  ESC - Exit")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return
        
        frame_count = 0
        saved_count = 0
        fps = 0
        last_time = time.time()
        
        try:
            while True:
                frame = self.pull_frame()
                if frame is None:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                
                frame_count += 1
                display_frame = frame.copy()
                
                # FPS calculation
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # Adaptive YOLO skipping
                self.frame_idx += 1
                any_calibrated = any(state["calibrated"] for state in self.homography_targets.values())
                mode_every_n = self.yolo_every_n_tracking if any_calibrated else self.yolo_every_n_search
                run_yolo = (self.frame_idx % mode_every_n == 0)
                
                # Safety: force YOLO if nothing detected
                if not any(b is not None for b in self.last_bboxes.values()):
                    self.no_det_frames += 1
                else:
                    self.no_det_frames = 0
                
                if self.no_det_frames > 10:
                    run_yolo = True
                    self.frame_idx = 0
                
                # YOLO detection for homography objects
                if run_yolo:
                    bboxes, confidences = self.detect_objects(frame)
                    self.last_bboxes = bboxes
                    self.last_confidences = confidences
                else:
                    bboxes = self.last_bboxes
                    confidences = self.last_confidences
                
                # ArUco detection (ALWAYS runs - independent!)
                aruco_corners, aruco_ids = self.detect_aruco_full_frame(frame)
                
                y_info_offset = 30
                tracked_count = 0
                
                # Process homography objects
                for obj_name, bbox in bboxes.items():
                    if bbox is None:
                        continue
                    
                    state = self.homography_targets[obj_name]
                    display_frame = self.draw_detection_box(display_frame, obj_name, bbox, is_homography=True)
                    
                    # Auto-calibration
                    if not state["calibrated"]:
                        conf = confidences.get(obj_name, 0.0)
                        if self.should_auto_calibrate(obj_name, bbox, conf, frame.shape):
                            if not self.accumulate_and_maybe_calibrate(frame, obj_name, bbox):
                                continue
                        else:
                            continue
                    
                    # Tracking
                    if state["calibrated"]:
                        corners = self.track_object_homography(frame, obj_name, bbox)
                        
                        if corners is not None:
                            rvec_raw, tvec_raw = self.estimate_pose_pnp(state["object_points_3d"], corners)
                            
                            if rvec_raw is not None and tvec_raw is not None:
                                # EMA smoothing
                                alpha = state["ema_alpha"]
                                if state["rvec_smooth"] is None:
                                    state["rvec_smooth"] = rvec_raw.copy()
                                    state["tvec_smooth"] = tvec_raw.copy()
                                else:
                                    state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                                    state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                                
                                rvec_final = state["rvec_smooth"]
                                tvec_final = state["tvec_smooth"]
                                
                                # Publish TF
                                self.publish_tf(obj_name, rvec_final, tvec_final)
                                
                                # Draw
                                display_frame = self.draw_tracked_corners(display_frame, obj_name, corners, is_homography=True)
                                display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final, is_homography=True)
                                y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, rvec_final, tvec_final, is_homography=True)
                                tracked_count += 1
                
                # Process ArUco objects (INDEPENDENT)
                if aruco_corners is not None and aruco_ids is not None:
                    for obj_name, aruco_state in self.aruco_targets.items():
                        target_id = aruco_state["aruco_id"]
                        
                        if target_id not in aruco_ids:
                            continue
                        
                        idx = np.where(aruco_ids == target_id)[0][0]
                        corners = aruco_corners[idx][0]
                        
                        # Estimate pose
                        rvec_raw, tvec_raw = self.estimate_pose_pnp(aruco_state["object_points_3d"], corners)
                        
                        if rvec_raw is not None and tvec_raw is not None:
                            # EMA smoothing
                            alpha = aruco_state["ema_alpha"]
                            if aruco_state["rvec_smooth"] is None:
                                aruco_state["rvec_smooth"] = rvec_raw.copy()
                                aruco_state["tvec_smooth"] = tvec_raw.copy()
                            else:
                                aruco_state["rvec_smooth"] = (1 - alpha) * aruco_state["rvec_smooth"] + alpha * rvec_raw
                                aruco_state["tvec_smooth"] = (1 - alpha) * aruco_state["tvec_smooth"] + alpha * tvec_raw
                            
                            rvec_final = aruco_state["rvec_smooth"]
                            tvec_final = aruco_state["tvec_smooth"]
                            
                            # Publish TF
                            self.publish_tf(obj_name, rvec_final, tvec_final)
                            
                            # Draw
                            display_frame = self.draw_tracked_corners(display_frame, obj_name, corners, is_homography=False)
                            display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final, is_homography=False)
                            y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, rvec_final, tvec_final, is_homography=False)
                            tracked_count += 1
                
                # UI
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cue = "SEARCH" if not any_calibrated else f"TRACK (skip:{mode_every_n})"
                color = (0, 255, 255) if not any_calibrated else (0, 255, 0)
                cv2.putText(display_frame, cue, (display_frame.shape[1]-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                if tracked_count > 0:
                    cv2.putText(display_frame, f"TRACKING {tracked_count}/5 + TF", (10, display_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('ROS2 Hybrid Pose Est', display_frame)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:
                    print("\nExiting...")
                    break
                
                elif key == ord('r'):
                    for obj_name in self.homography_targets:
                        state = self.homography_targets[obj_name]
                        state["calibrated"] = False
                        state["ref_image"] = None
                        state["rvec_smooth"] = None
                        state["tvec_smooth"] = None
                        state["calib_buffer"].clear()
                    print("✓ All calibrations reset.")
                
                elif key == ord('s'):
                    filename = f"hybrid_result_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    print(f"✓ Saved: {filename}")
        
        finally:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("SETUP - ROS2 HYBRID")
    print("="*60)
    
    yolo_model_path = 'runs/detect/yolov8n_detect_V1/weights/best.pt'
    
    # ROS2 init
    rclpy.init()
    
    # Create node
    estimator = HybridPoseEstimator(yolo_model_path)
    
    print("\n--- Object Configurations ---")
    for obj_name, state in estimator.homography_targets.items():
        print(f"-> {obj_name}: {state['width_mm']}x{state['height_mm']}mm (homography, buffer:5)")
    for obj_name, state in estimator.aruco_targets.items():
        print(f"-> {obj_name}: ArUco ID {state['aruco_id']} (aruco, independent)")
    print("="*60 + "\n")
    
    # Run
    try:
        estimator.run()
    finally:
        estimator.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()