"""
Combined Pose Estimation System - Manual Calibration (IMPROVED)
Tracks all 5 objects using both Homography and ArUco methods
- Homography: card_game, circuit_board, notebook (STRICTER QUALITY CONTROL)
- ArUco: estop, phone
- Manual calibration with stationary detection
- Independent parallel tracking after calibration
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ==============================================================================
# ARUCO CONFIG
# ==============================================================================
ARUCO_DICT = cv2.aruco.DICT_4X4_50

ARUCO_IDS = {
    "phone": 0,
    "estop": 1,
}

ARUCO_MARKER_SIZES = {
    "phone": 50.0,   # mm
    "estop": 32.0,  # mm
}

# ==============================================================================
# OBJECT CONFIGS
# ==============================================================================
OBJECT_CONFIGS = {
    # Homography objects
    "card_game": {
        "label_substring": "card_game",
        "width_mm": 93.0,
        "height_mm": 115.0,
        "axis_color": (0, 255, 0),       # Green
        "method": "homography",
        "ema_alpha": 0.2,
    },
    "circuit_board": {
        "label_substring": "circuit_board",
        "width_mm": 210.0,
        "height_mm": 210.0,
        "axis_color": (255, 0, 0),       # Blue
        "method": "homography",
        "ema_alpha": 0.2,
    },
    "notebook": {
        "label_substring": "notebook",
        "width_mm": 147.0,
        "height_mm": 209.0,
        "axis_color": (0, 0, 255),       # Red
        "method": "homography",
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
# QUALITY THRESHOLDS (STRICTER - LIKE SUITABILITY TESTER)
# ==============================================================================
STATIONARY_TIME_THRESHOLD = 3.0   # seconds
STATIONARY_MOVEMENT_THRESHOLD = 15  # pixels
YOLO_CONFIDENCE_THRESHOLD = 0.5   # Confidence gate for stationary check

# Homography quality thresholds (stricter like the tester)
MIN_REF_KP = 80          # Minimum keypoints for reference calibration
MIN_TRACK_KP = 15        # Minimum keypoints in current frame for tracking
MIN_GOOD_MATCHES = 20    # Minimum good matches for homography
MIN_INLIER_RATIO = 0.40  # Minimum inlier ratio for RANSAC


class CombinedPoseEstimator:
    def __init__(self, yolo_model_path, camera_index=1):
        self.cap = None
        self.camera_index = camera_index

        # Load YOLO model
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

        # Frame skipping for YOLO
        self.frame_idx = 0
        self.yolo_every_n = 3

        # Build per-object state
        self.targets = {}

        for obj_name, cfg in OBJECT_CONFIGS.items():
            label_sub = cfg["label_substring"].lower()
            class_id = next(
                (cid for cid, cname in self.class_names.items()
                 if label_sub in cname.lower()),
                None,
            )
            
            needs_calibration = (cfg["method"] == "homography")
            
            state = {
                "class_id": class_id,
                "method": cfg["method"],
                "axis_color": cfg["axis_color"],
                "ema_alpha": cfg.get("ema_alpha", 0.2),
                "rvec_smooth": None,
                "tvec_smooth": None,
                "calibrated": (not needs_calibration),
                "is_tracked": False,
                # Stationary detection
                "stationary_start_time": None,
                "stationary_bbox_center": None,
                "ready_to_calibrate": False,
                "last_bbox": None,
                "last_conf": 0.0,
                # Quality metrics
                "last_inlier_ratio": 0.0,
                "last_good_matches": 0,
            }

            if cfg["method"] == "homography":
                width_mm = cfg["width_mm"]
                height_mm = cfg["height_mm"]
                half_w = width_mm / 2.0
                half_h = height_mm / 2.0

                state.update({
                    "object_points_3d": np.array([
                        [-half_w,  half_h, 0], [ half_w,  half_h, 0],
                        [ half_w, -half_h, 0], [-half_w, -half_h, 0],
                    ], dtype=np.float32),
                    "ref_image": None,
                    "ref_keypoints": None,
                    "ref_descriptors": None,
                    "ref_bbox": None,
                    "ref_kp_count": 0,
                })

            elif cfg["method"] == "aruco":
                marker_size = cfg["marker_size_mm"]
                half_size = marker_size / 2.0

                state.update({
                    "aruco_id": cfg["aruco_id"],
                    "marker_size_mm": marker_size,
                    "object_points_3d": np.array([
                        [-half_size,  half_size, 0], [ half_size,  half_size, 0],
                        [ half_size, -half_size, 0], [-half_size, -half_size, 0],
                    ], dtype=np.float32),
                })

            self.targets[obj_name] = state

        # Camera parameters
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DIST_COEFFS
        print("✓ Loaded camera parameters")

        # ORB for homography - MORE SENSITIVE for better features
        self.orb = cv2.ORB_create(
            nfeatures=5000,        # Even more features (was 3000)
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=5        # More sensitive = more keypoints
        )
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print(f"✓ ORB initialized (5000 features, sensitive mode)")

        # Shared feature state for parallel Homography tracking
        self.current_frame_keypoints = None
        self.current_frame_descriptors = None

    def start_camera(self):
        """Start webcam"""
        print(f"Starting webcam (index {self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_index}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✓ Webcam started: 640x480")
        return True

    def pull_frame(self):
        """Read frame from webcam"""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def detect_objects(self, frame):
        """Run YOLO detection"""
        results = self.yolo_model(frame, verbose=False, imgsz=self.yolo_imgsz, conf=0.3, iou=0.5)
        result = results[0]

        bboxes = {name: None for name in self.targets}
        confidences = {name: 0.0 for name in self.targets}

        if result.boxes is None or len(result.boxes) == 0:
            return bboxes, confidences

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            for name, state in self.targets.items():
                if state["class_id"] is None:
                    continue
                if class_id == state["class_id"] and confidence > confidences[name]:
                    bboxes[name] = [x1, y1, x2, y2]
                    confidences[name] = confidence

        return bboxes, confidences

    def check_stationary(self, obj_name, bbox, current_time):
        """Check if object has been stationary for threshold time"""
        state = self.targets[obj_name]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if state["stationary_bbox_center"] is None:
            state["stationary_bbox_center"] = (center_x, center_y)
            state["stationary_start_time"] = current_time
            state["ready_to_calibrate"] = False
            return False
        
        prev_cx, prev_cy = state["stationary_bbox_center"]
        distance = np.sqrt((center_x - prev_cx)**2 + (center_y - prev_cy)**2)
        
        if distance > STATIONARY_MOVEMENT_THRESHOLD:
            state["stationary_bbox_center"] = (center_x, center_y)
            state["stationary_start_time"] = current_time
            state["ready_to_calibrate"] = False
            return False
        
        elapsed = current_time - state["stationary_start_time"]
        if elapsed >= STATIONARY_TIME_THRESHOLD:
            state["ready_to_calibrate"] = True
            return True
        
        return False

    def calibrate_reference(self, frame, obj_name, bbox):
        """Calibrate reference for homography object (STRICT - like tester)"""
        state = self.targets[obj_name]
        print(f"\n{'='*60}")
        print(f"CALIBRATING: {obj_name.upper()}")
        print('='*60)

        x1, y1, x2, y2 = bbox
        
        # Pad bbox to include more of the object
        pad = 10
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        num_kp = len(kp) if kp is not None else 0
        print(f"Reference keypoints detected: {num_kp}")

        if des is None or num_kp < MIN_REF_KP:
            print(f"✗ NOT ENOUGH FEATURES for {obj_name}: {num_kp} (need ≥ {MIN_REF_KP})")
            print(f"✗ Try repositioning or use an object with more texture")
            state["stationary_bbox_center"] = None
            state["stationary_start_time"] = None
            state["ready_to_calibrate"] = False
            return False

        state["ref_image"] = gray
        state["ref_keypoints"] = kp
        state["ref_descriptors"] = des
        state["ref_bbox"] = [x1, y1, x2, y2]
        state["ref_kp_count"] = num_kp

        state["rvec_smooth"] = None
        state["tvec_smooth"] = None
        state["calibrated"] = True
        state["ready_to_calibrate"] = False
        state["stationary_bbox_center"] = None
        state["stationary_start_time"] = None

        print(f"✓ CALIBRATION COMPLETE for {obj_name.upper()}")
        print(f"  - Reference KP: {num_kp}")
        print('='*60 + '\n')
        return True

    def track_object_homography_independent(self, obj_name, keypoints, descriptors):
        """Track using homography - STRICT quality requirements"""
        state = self.targets[obj_name]

        if not state.get("calibrated") or state["ref_descriptors"] is None:
            return None

        # Strict keypoint requirements
        if keypoints is None or descriptors is None or len(keypoints) < MIN_TRACK_KP:
            return None

        matches = self.bf_matcher.knnMatch(state["ref_descriptors"], descriptors, k=2)

        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Strict match requirements
        if len(good_matches) < MIN_GOOD_MATCHES:
            return None

        src_pts = np.float32([state["ref_keypoints"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        if H is None or mask is None:
            return None

        inliers = mask.ravel().tolist()
        inlier_ratio = sum(inliers) / len(inliers)
        
        # Strict inlier ratio requirement
        if inlier_ratio < MIN_INLIER_RATIO:
            return None

        h_ref, w_ref = state["ref_image"].shape
        ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)
        corners_frame = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)

        # Store quality metrics for display
        state["last_inlier_ratio"] = inlier_ratio
        state["last_good_matches"] = len(good_matches)

        return corners_frame

    def detect_aruco_independent(self, frame):
        """Detect all ArUco markers in entire frame"""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_aruco_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray_frame)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None or len(ids) == 0:
            return {}
        
        detections = {}
        for obj_name, state in self.targets.items():
            if state["method"] != "aruco":
                continue
            target_id = state["aruco_id"]
            if target_id in ids.flatten():
                idx = np.where(ids.flatten() == target_id)[0][0]
                detections[obj_name] = corners[idx][0]
        
        return detections

    def estimate_pose_pnp(self, obj_name, corners_2d):
        """Estimate 6DOF pose"""
        state = self.targets[obj_name]
        image_points = corners_2d.reshape(-1, 1, 2).astype(np.float32)

        if state["method"] == "aruco":
            success, rvec, tvec = cv2.solvePnP(
                state["object_points_3d"], image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                return None, None
            return rvec, tvec
        else:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                state["object_points_3d"], image_points,
                self.camera_matrix, self.dist_coeffs,
                reprojectionError=8.0, flags=cv2.SOLVEPNP_ITERATIVE
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

    def draw_detection_box(self, frame, obj_name, bbox):
        """Draw YOLO detection box"""
        x1, y1, x2, y2 = bbox
        state = self.targets[obj_name]
        
        color = (150, 255, 150)
        
        if state["method"] == "homography":
            status = "[CAL]" if state.get("calibrated") else "[UNCAL]"
            label = f"{obj_name} {status}"
        else:
            label = f"{obj_name} (ArUco)"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return frame

    def draw_tracked_corners(self, frame, obj_name, corners):
        """Draw tracked corners"""
        corners_int = corners.astype(int)
        color = self.targets[obj_name]["axis_color"]
        cv2.polylines(frame, [corners_int], True, color, 3)
        return frame

    def draw_3d_axes(self, frame, obj_name, rvec, tvec, length=50):
        """Draw 3D axes"""
        color = self.targets[obj_name]["axis_color"]
        
        axis_points_3d = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
        axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        origin = tuple(axis_points_2d[0])
        
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3, tipLength=0.3)
        cv2.circle(frame, origin, 8, color, -1)
        
        return frame

    def draw_pose_info(self, frame, obj_name, y_offset_start, rvec_final, tvec_final):
        """Draw pose info WITH QUALITY METRICS"""
        roll, pitch, yaw = self.rotation_to_euler(rvec_final)
        state = self.targets[obj_name]
        text_color = state["axis_color"]
        y_offset = y_offset_start
        
        cv2.putText(frame, f"--- {obj_name.upper()} ---", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        y_offset += 20
        
        cv2.putText(frame, f"Pos: [{tvec_final[0,0]:.0f}, {tvec_final[1,0]:.0f}, {tvec_final[2,0]:.0f}]mm",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 18
        
        cv2.putText(frame, f"R/P/Y: [{roll:.1f}, {pitch:.1f}, {yaw:.1f}]deg",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 18
        
        # Quality metrics for homography objects
        if state["method"] == "homography" and state.get("calibrated"):
            if "ref_kp_count" in state:
                cv2.putText(frame, f"Ref KP: {state['ref_kp_count']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y_offset += 15
            
            if state.get("last_good_matches", 0) > 0:
                cv2.putText(frame, f"Matches: {state['last_good_matches']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                y_offset += 15
            
            if state.get("last_inlier_ratio", 0) > 0:
                ratio = state['last_inlier_ratio']
                ratio_color = (0, 255, 0) if ratio >= 0.5 else (0, 165, 255) if ratio >= 0.4 else (0, 0, 255)
                cv2.putText(frame, f"Inlier: {ratio:.2f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ratio_color, 1)
                y_offset += 15
        
        y_offset += 10
        return y_offset

    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("COMBINED POSE ESTIMATION SYSTEM (IMPROVED)")
        print("STRICTER HOMOGRAPHY QUALITY CONTROL")
        print("AUTO-CALIBRATION ENABLED")
        print("="*60)
        print("\nWorkflow:")
        print("  - Hold object steady for 3 seconds")
        print("  - System will auto-calibrate with strict quality checks")
        print("  - Tracking begins immediately after calibration")
        print("\nControls:")
        print("  'r' - Reset all calibrations")
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
                
                # Centralized ORB feature calculation
                any_homography_calibrated = any(
                    s["method"] == "homography" and s["calibrated"]
                    for s in self.targets.values()
                )
                
                if any_homography_calibrated:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.current_frame_keypoints, self.current_frame_descriptors = self.orb.detectAndCompute(gray_frame, None)
                else:
                    self.current_frame_keypoints, self.current_frame_descriptors = None, None

                frame_count += 1
                display_frame = frame.copy()
                current_time = time.time()
                
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # YOLO detection
                self.frame_idx += 1
                run_yolo = (self.frame_idx % self.yolo_every_n == 0)
                
                if run_yolo:
                    current_bboxes, confidences = self.detect_objects(frame)
                    for name, bbox in current_bboxes.items():
                        self.targets[name]["last_bbox"] = bbox
                        self.targets[name]["last_conf"] = confidences[name]
                else:
                    current_bboxes = {name: state.get("last_bbox") for name, state in self.targets.items()}
                    confidences = {name: state.get("last_conf", 0.0) for name, state in self.targets.items()}

                # ArUco detection
                aruco_detections = self.detect_aruco_independent(frame)
                
                y_info_offset = 30
                tracked_count = 0
                
                # Draw bboxes and handle calibration
                for obj_name, state in self.targets.items():
                    state["is_tracked"] = False
                    bbox = state["last_bbox"]
                    conf = confidences.get(obj_name, 0.0)
                    
                    # Always draw bbox if available
                    if bbox is not None:
                        display_frame = self.draw_detection_box(display_frame, obj_name, bbox)

                    # AUTO-CALIBRATION logic (homography only)
                    if state["method"] == "homography" and not state["calibrated"]:
                        if bbox is not None and conf > YOLO_CONFIDENCE_THRESHOLD:
                            is_stationary = self.check_stationary(obj_name, bbox, current_time)
                            
                            # Auto-calibrate when stationary
                            if is_stationary and state["ready_to_calibrate"]:
                                print(f"\n[AUTO-CALIBRATING {obj_name.upper()}]")
                                success = self.calibrate_reference(frame, obj_name, bbox)
                                if not success:
                                    # Reset to try again
                                    state["stationary_bbox_center"] = None
                                    state["stationary_start_time"] = None
                                    state["ready_to_calibrate"] = False
                        
                        elif bbox is None or conf <= YOLO_CONFIDENCE_THRESHOLD:
                            state["stationary_bbox_center"] = None
                            state["stationary_start_time"] = None
                            state["ready_to_calibrate"] = False

                # Homography tracking
                for obj_name, state in self.targets.items():
                    if state["method"] == "homography" and state["calibrated"]:
                        corners = self.track_object_homography_independent(
                            obj_name, 
                            self.current_frame_keypoints, 
                            self.current_frame_descriptors
                        )
                        rvec_raw, tvec_raw = None, None
                        if corners is not None:
                            rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                        
                        if rvec_raw is not None:
                            alpha = state["ema_alpha"]
                            if state["rvec_smooth"] is None:
                                state["rvec_smooth"] = rvec_raw.copy()
                                state["tvec_smooth"] = tvec_raw.copy()
                            else:
                                state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                                state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                                
                            state["is_tracked"] = True
                            
                            display_frame = self.draw_tracked_corners(display_frame, obj_name, corners)
                            display_frame = self.draw_3d_axes(display_frame, obj_name, state["rvec_smooth"], state["tvec_smooth"])
                            y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, state["rvec_smooth"], state["tvec_smooth"])
                            tracked_count += 1
                        
                        elif state["rvec_smooth"] is not None:
                            txt = f"{obj_name.upper()}: LOST"
                            cv2.putText(display_frame, txt, (10, y_info_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            y_info_offset += 25
                    
                # ArUco tracking
                for obj_name, state in self.targets.items():
                    if state["method"] == "aruco":
                        corners = aruco_detections.get(obj_name)
                        
                        rvec_raw, tvec_raw = None, None
                        if corners is not None:
                            rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                            
                        if rvec_raw is not None:
                            alpha = state["ema_alpha"]
                            if state["rvec_smooth"] is None:
                                state["rvec_smooth"] = rvec_raw.copy()
                                state["tvec_smooth"] = tvec_raw.copy()
                            else:
                                state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                                state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                            
                            state["is_tracked"] = True
                            
                            display_frame = self.draw_tracked_corners(display_frame, obj_name, corners)
                            display_frame = self.draw_3d_axes(display_frame, obj_name, state["rvec_smooth"], state["tvec_smooth"])
                            y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, state["rvec_smooth"], state["tvec_smooth"])
                            tracked_count += 1
                        
                        elif state["rvec_smooth"] is not None:
                            txt = f"{obj_name.upper()}: LOST"
                            cv2.putText(display_frame, txt, (10, y_info_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            y_info_offset += 25

                # UI
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if tracked_count > 0:
                    cv2.putText(display_frame, f"TRACKING {tracked_count}/5 OBJECTS", 
                                (10, display_frame.shape[0]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Combined Pose Estimation - IMPROVED', display_frame)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:
                    print("\nExiting...")
                    break
                
                elif key == ord('r'):
                    for obj_name, state in self.targets.items():
                        if state["method"] == "homography":
                            state["calibrated"] = False
                            state["ref_image"] = None
                            state["ref_descriptors"] = None
                            state["stationary_bbox_center"] = None
                            state["stationary_start_time"] = None
                            state["ready_to_calibrate"] = False
                        
                        state["rvec_smooth"] = None
                        state["tvec_smooth"] = None
                        state["is_tracked"] = False
                        state["last_bbox"] = None
                        state["last_conf"] = 0.0
                        state["last_inlier_ratio"] = 0.0
                        state["last_good_matches"] = 0
                    
                    print("✓ All object states reset")
                
                elif key == ord('s'):
                    filename = f"improved_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    print(f"✓ Saved: {filename}")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    yolo_model_path = './runs/detect/yolov8n_detect_V2/weights/best.pt'
    camera_index = 1
    
    estimator = CombinedPoseEstimator(yolo_model_path, camera_index)
    
    print("\n--- Object Configurations ---")
    for obj_name, state in estimator.targets.items():
        method = state["method"]
        if method == "homography":
            w = state['object_points_3d'][1,0] * 2
            h = state['object_points_3d'][0,1] * 2
            print(f"→ {obj_name}: {w:.0f}x{h:.0f}mm ({method})")
        else:
            print(f"→ {obj_name}: ArUco ID {state['aruco_id']}, {state['marker_size_mm']}mm ({method})")
    print("="*60)
    print(f"Quality Thresholds: MIN_REF_KP={MIN_REF_KP}, MIN_GOOD_MATCHES={MIN_GOOD_MATCHES}, MIN_INLIER_RATIO={MIN_INLIER_RATIO}")
    print("="*60 + "\n")

    estimator.run()


if __name__ == "__main__":
    main()