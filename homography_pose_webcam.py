"""
Test Homography + PnP with Existing YOLO Model
Webcam Version for Laptop Testing
Tracking: game_box, repair_mat (screw mat), and notebook
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ==============================================================================
# OBJECT CONFIGS
# ==============================================================================
OBJECT_CONFIGS = {
    "repair_mat": {
        "label_substring": "repair_mat", 
        "width_mm": 89.0,                
        "height_mm": 144.0, 
        "axis_color": (0, 255, 0),       # Green
        "min_matches": 8,
        "min_inlier_ratio": 0.4,
        "ema_alpha": 0.2,
    },
    "game_box": {                
        "label_substring": "game_box", 
        "width_mm": 95.0,                
        "height_mm": 124.0, 
        "axis_color": (255, 0, 0),       # Blue
        "min_matches": 10,
        "min_inlier_ratio": 0.5,
        "ema_alpha": 0.2,
    },
    "notebook": {                
        "label_substring": "notebook", 
        "width_mm": 210.0,                
        "height_mm": 295.0, 
        "axis_color": (0, 0, 255),       # Red
        "min_matches": 10,
        "min_inlier_ratio": 0.5,
        "ema_alpha": 0.2,
    },
}
# ==============================================================================

class MultiObjectPoseEstimator:
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

        # Frame skipping state
        self.frame_idx = 0
        self.yolo_every_n_tracking = 3   # Check every 3 frames when tracking
        self.yolo_every_n_search = 1     # Check every frame when searching
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
                print(f"WARNING: '{obj_name}' class not found in model!")
            else:
                print(f"✓ {obj_name.capitalize()} class found: ID={class_id}")

            width_mm = cfg["width_mm"]
            height_mm = cfg["height_mm"]
            half_w = width_mm / 2.0
            half_h = height_mm / 2.0

            object_points_3d = np.array([
                [-half_w,  half_h, 0],   # TL (Top-Left)
                [ half_w,  half_h, 0],   # TR (Top-Right)
                [ half_w, -half_h, 0],   # BR (Bottom-Right)
                [-half_w, -half_h, 0],   # BL (Bottom-Left)
            ], dtype=np.float32)

            self.targets[obj_name] = {
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
                "ref_bbox_size": None,
                "calibrated": False,
                "rvec_smooth": None,
                "tvec_smooth": None,
                "calib_buffer": [],
                "calib_buffer_size": 3, 
            }

        # Camera parameters
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DIST_COEFFS

        print("✓ Loaded camera parameters from camera_params.py for PnP.")

        # ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=1500, 
            fastThreshold=12 
        )
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print(f"✓ ORB initialized (1500 features).")

    def start_camera(self):
        """Start webcam capture"""
        print(f"Starting webcam (index {self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open camera {self.camera_index}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"✓ Webcam started: 640x480")
        return True

    def pull_frame(self):
        """Read frame from webcam"""
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame

    def detect_objects(self, frame):
        """Run YOLO detection"""
        results = self.yolo_model(
            frame, 
            verbose=False,
            imgsz=self.yolo_imgsz, 
            conf=0.8,
            iou=0.5
        )
        result = results[0]

        bboxes = {name: None for name in self.targets}
        best_conf = {name: 0.0 for name in self.targets}

        if result.boxes is None or len(result.boxes) == 0:
            return bboxes, best_conf

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            for name, state in self.targets.items():
                if state["class_id"] is None:
                    continue
                if class_id == state["class_id"] and confidence > best_conf[name]:
                    bboxes[name] = [x1, y1, x2, y2]
                    best_conf[name] = confidence

        return bboxes, best_conf

    def calibrate_reference(self, frame, obj_name, bbox):
        """Calibrate reference image for an object"""
        state = self.targets[obj_name]
        print("\n" + "="*60)
        print(f"CALIBRATING FROM BEST CANDIDATE: {obj_name.upper()}")
        print("="*60)

        x1, y1, x2, y2 = bbox
        
        if x1 == 0 and y1 == 0 and x2 == frame.shape[1] and y2 == frame.shape[0]:
            roi = frame
        else:
            roi = frame[y1:y2, x1:x2]

        state["ref_image"] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        state["ref_bbox_size"] = (roi.shape[1], roi.shape[0])
        state["ref_keypoints"], state["ref_descriptors"] = self.orb.detectAndCompute(
            state["ref_image"], None
        )

        if state["ref_descriptors"] is None or len(state["ref_keypoints"]) < 10:
            print("ERROR: Not enough features in this candidate! Retrying.")
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
        """Check if object is suitable for auto-calibration"""
        if confidence < 0.8:
            return False

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        H, W = frame_shape[:2]
        area = w * h
        frame_area = W * H

        if area < 0.02 * frame_area:
            return False

        if obj_name == "repair_mat":
            if area < 0.015 * frame_area:
                return False

        return True

    def accumulate_and_maybe_calibrate(self, frame, obj_name, bbox):
        """Accumulate calibration candidates and calibrate when buffer is full"""
        state = self.targets[obj_name]
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
        """Track object using homography"""
        state = self.targets[obj_name]

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

        src_pts = np.float32(
            [state["ref_keypoints"][m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [keypoints[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None

        inliers = mask.ravel().tolist()
        inlier_ratio = sum(inliers) / len(inliers)
        if inlier_ratio < min_inlier_ratio:
            return None

        h_ref, w_ref = state["ref_image"].shape
        ref_corners = np.array(
            [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32
        ).reshape(-1, 1, 2)

        corners_roi = cv2.perspectiveTransform(ref_corners, H)
        corners_roi = corners_roi.reshape(-1, 2)

        corners_frame = corners_roi.copy()
        corners_frame[:, 0] += x1
        corners_frame[:, 1] += y1

        return corners_frame

    def estimate_pose_pnp(self, obj_name, corners_2d):
        """Estimate 6DOF pose using PnP"""
        state = self.targets[obj_name]
        
        image_points = corners_2d.reshape(-1, 1, 2).astype(np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            state["object_points_3d"], 
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
    
    def draw_detection_box(self, frame, obj_name, bbox):
        """Draw detection bounding box"""
        x1, y1, x2, y2 = bbox
        state = self.targets[obj_name]
        color = (0, 255, 0) if state["calibrated"] else (255, 0, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{obj_name.capitalize()} (YOLO)", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
    
    def draw_tracked_corners(self, frame, obj_name, corners):
        """Draw tracked corners"""
        corners_int = corners.astype(int)
        color = self.targets[obj_name]["axis_color"] 
        cv2.polylines(frame, [corners_int], True, color, 3)
        return frame
    
    def draw_3d_axes(self, frame, obj_name, rvec, tvec, length=100):
        """Draw 3D coordinate axes"""
        color = self.targets[obj_name]["axis_color"]
        
        axis_points_3d = np.float32([
            [0, 0, 0],       
            [length, 0, 0],  
            [0, length, 0],  
            [0, 0, length]   
        ])
        
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        origin = tuple(axis_points_2d[0])
        
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3, tipLength=0.3)  # X-Red 
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3, tipLength=0.3)  # Y-Green 
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3, tipLength=0.3)  # Z-Blue 
        
        cv2.circle(frame, origin, 8, color, -1)
        
        return frame

    def draw_pose_info(self, frame, obj_name, y_offset_start, rvec_final, tvec_final):
        """Draw pose information on frame"""
        roll, pitch, yaw = self.rotation_to_euler(rvec_final)
        distance_z = tvec_final[2, 0]
        
        text_color = self.targets[obj_name]["axis_color"]
        y_offset = y_offset_start
        
        cv2.putText(frame, f"--- {obj_name.upper()} ---", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        y_offset += 25
        
        cv2.putText(frame, f"X,Y,Z Pos: [{tvec_final[0, 0]:.0f}, {tvec_final[1, 0]:.0f}, {distance_z:.0f}]mm", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Roll (X): {roll:.1f}deg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        cv2.putText(frame, f"Pitch (Y): {pitch:.1f}deg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"Yaw (Z): {yaw:.1f}deg", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += 25
        
        return y_offset

    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("YOLO DETECTION + HOMOGRAPHY + PnP TEST (MULTI-OBJECT)")
        print("WEBCAM VERSION FOR LAPTOP TESTING")
        print("TRACKING: repair_mat, game_box, notebook")
        print("="*60)
        print("\nControls:")
        print("  'r' - Reset ALL calibrations")
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
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count
                    frame_count = 0
                    last_time = current_time
                
                # Adaptive YOLO Skipping
                self.frame_idx += 1
                
                any_calibrated = any(state["calibrated"] for state in self.targets.values())
                mode_every_n = self.yolo_every_n_tracking if any_calibrated else self.yolo_every_n_search
                
                run_yolo = (self.frame_idx % mode_every_n == 0)
                
                if not any(b is not None for b in self.last_bboxes.values()):
                    self.no_det_frames += 1
                else:
                    self.no_det_frames = 0
                
                if self.no_det_frames > 10:
                    run_yolo = True
                    self.frame_idx = 0
                
                if run_yolo:
                    bboxes, confidences = self.detect_objects(frame)
                    self.last_bboxes = bboxes
                    self.last_confidences = confidences
                else:
                    bboxes = self.last_bboxes
                    confidences = self.last_confidences
                
                y_info_offset = 30
                tracked_objects_count = 0
                
                detected_objects = [
                    obj_name for obj_name, bbox in bboxes.items() if bbox is not None
                ]
                
                # Per-object loop
                for obj_name in detected_objects:
                    bbox = bboxes[obj_name]
                    state = self.targets[obj_name]
                    
                    display_frame = self.draw_detection_box(display_frame, obj_name, bbox)
                    
                    # Autocalibration 
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
                            rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                            
                            if rvec_raw is not None and tvec_raw is not None:
                                # EMA Temporal Smoothing
                                alpha = state.get("ema_alpha", 0.2) 
                                if state["rvec_smooth"] is None:
                                    state["rvec_smooth"] = rvec_raw.copy()
                                    state["tvec_smooth"] = tvec_raw.copy()
                                else:
                                    state["rvec_smooth"] = (1 - alpha) * state["rvec_smooth"] + alpha * rvec_raw
                                    state["tvec_smooth"] = (1 - alpha) * state["tvec_smooth"] + alpha * tvec_raw
                                
                                rvec_final = state["rvec_smooth"]
                                tvec_final = state["tvec_smooth"]

                                display_frame = self.draw_tracked_corners(display_frame, obj_name, corners)
                                display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final)
                                
                                y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, rvec_final, tvec_final)
                                tracked_objects_count += 1
                                
                            else:
                                y_info_offset += 25
                                cv2.putText(display_frame, f"Pose RANSAC failed for {obj_name}", (10, y_info_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        else:
                            y_info_offset += 25
                            cv2.putText(display_frame, f"Homography/Inlier failed for {obj_name}", (10, y_info_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # UI/FPS
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cue = "Search (YOLO)" if run_yolo else f"Tracking (SKIP {mode_every_n})"
                color = (0, 255, 0) if not run_yolo else (0, 255, 255)
                cv2.putText(display_frame, cue, (display_frame.shape[1]-250, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if tracked_objects_count > 0:
                    cv2.putText(display_frame, f"TRACKING {tracked_objects_count} OBJECT(S) OK", (10, display_frame.shape[0]-50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv2.imshow('Webcam Pose Estimation', display_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                
                elif key == ord('r'):  # Reset
                    for obj_name in self.targets:
                        state = self.targets[obj_name]
                        state["calibrated"] = False
                        state["ref_image"] = None
                        state["rvec_smooth"] = None 
                        state["tvec_smooth"] = None
                        state["calib_buffer"].clear()
                    print("✓ All calibrations reset.")
                
                elif key == ord('s'):  # Save
                    filename = f"test_result_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    print(f"✓ Saved: {filename}")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("SETUP")
    print("="*60)
    print("✓ Using camera calibration from camera_params.py")
    
    # UPDATE THIS PATH TO YOUR TRAINED MODEL
    yolo_model_path = 'runs/detect/yolov8n_detect_V1/weights/best.pt'
    
    # Camera index (usually 0 for built-in webcam)
    camera_index = 1
    
    estimator = MultiObjectPoseEstimator(yolo_model_path, camera_index)
    
    print("\n--- Object Configurations ---")
    for obj_name, state in estimator.targets.items():
        print(f"-> {obj_name.capitalize()}: {state['width_mm']}x{state['height_mm']}mm "
              f"(Buffer: {state['calib_buffer_size']}, EMA: {state['ema_alpha']})")
    print("="*60 + "\n")

    estimator.run()


if __name__ == "__main__":
    main()