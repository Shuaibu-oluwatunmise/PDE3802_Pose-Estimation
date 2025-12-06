import cv2
import numpy as np
from ultralytics import YOLO
import time
from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ============================================================================
# CONFIGURATION
# ============================================================================
YOLO_MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'

# Object specifications (real-world dimensions in meters)
OBJECT_SPECS = {
    "card_game": {
        "width_m": 0.093,   # 93mm
        "height_m": 0.115,  # 115mm
    }
}

TARGET_OBJECT = "card_game"  # What we're tracking

# Detection parameters
MIN_DETECTION_CONFIDENCE = 0.70
STABLE_FRAMES_NEEDED = 5  # How many stable frames before capturing reference
BBOX_STABILITY_THRESH = 15  # pixels

# Feature detection
MIN_FEATURES = 20
ORB_FEATURES = 2000

# Pose estimation
MIN_MATCH_COUNT = 15
MIN_PNP_POINTS = 8
MIN_PNP_INLIERS = 8
REPROJ_ERROR_THRESH = 3.0

# Visualization
DRAW_BBOX = True
DRAW_MATCHES = False  # Set True to see feature matches
DRAW_INLIERS = True

# Camera selection
CAMERA_INDEX = 1  # Change to 1 for second camera

UNDISTORT = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_calibration():
    """Load camera calibration data from imported params."""
    try:
        K = np.array(CAMERA_MATRIX, dtype=np.float32)
        dist = np.array(DIST_COEFFS, dtype=np.float32) if DIST_COEFFS is not None else None
        print("Camera calibration loaded successfully!")
        return K, dist
    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")
        print("Using default camera matrix. Results may be inaccurate.")
        # Default rough calibration for typical webcam
        K = np.array([[800, 0, 320],
                      [0, 800, 240],
                      [0, 0, 1]], dtype=np.float32)
        dist = None
        return K, dist


def bbox_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two bboxes."""
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2
    
    # Intersection area
    x_left = max(x1_tl, x2_tl)
    y_top = max(y1_tl, y2_tl)
    x_right = min(x1_br, x2_br)
    y_bottom = min(y1_br, y2_br)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_br - x1_tl) * (y1_br - y1_tl)
    area2 = (x2_br - x2_tl) * (y2_br - y2_tl)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def is_blurry(image, threshold=100):
    """Check if image is too blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def serialize_keypoints(keypoints):
    """Convert cv2.KeyPoint list into arrays."""
    kp_xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    kp_meta = np.array([
        (kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ], dtype=np.float32)
    return kp_xy, kp_meta


def reconstruct_keypoints(kp_xy, kp_meta):
    """Reconstruct cv2.KeyPoint list from arrays."""
    keypoints = []
    for (x, y), (size, angle, resp, octv, cid) in zip(kp_xy, kp_meta):
        kp = cv2.KeyPoint(float(x), float(y), float(size),
                          float(angle), float(resp), int(octv), int(cid))
        keypoints.append(kp)
    return keypoints


# ============================================================================
# MAIN TRACKER CLASS
# ============================================================================

class CardGameTracker:
    def __init__(self):
        # Load models and calibration
        print("Loading YOLO model...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        print("Model loaded successfully!")
        
        self.K, self.dist = load_calibration()
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Reference data (will be captured automatically)
        self.ref_image = None
        self.ref_kp = None
        self.ref_des = None
        self.plane_ref = None  # 3D coordinates of reference keypoints
        self.ref_bbox = None
        
        # Object dimensions
        self.obj_width = OBJECT_SPECS[TARGET_OBJECT]["width_m"]
        self.obj_height = OBJECT_SPECS[TARGET_OBJECT]["height_m"]
        
        # Tracking state
        self.has_reference = False
        self.stable_bbox_buffer = []
        self.prev_tvec = None
        self.prev_rvec = None
        
        # Stats
        self.frame_count = 0
        self.reference_capture_frame = None
        
    def is_bbox_stable(self, bbox):
        """Check if bbox has been stable for N frames."""
        self.stable_bbox_buffer.append(bbox)
        
        if len(self.stable_bbox_buffer) > STABLE_FRAMES_NEEDED:
            self.stable_bbox_buffer.pop(0)
        
        if len(self.stable_bbox_buffer) < STABLE_FRAMES_NEEDED:
            return False
        
        # Check stability: all bboxes should have high IoU with first
        first_bbox = self.stable_bbox_buffer[0]
        for bbox in self.stable_bbox_buffer[1:]:
            if bbox_iou(first_bbox, bbox) < 0.85:
                return False
        
        return True
    
    def create_reference(self, frame, bbox):
        """Create reference from detected object in frame."""
        x1, y1, x2, y2 = bbox
        
        # Crop the detected region
        crop = frame[y1:y2, x1:x2].copy()
        
        # Check if crop is good quality
        if is_blurry(crop, threshold=100):
            print("Frame too blurry for reference.")
            return False
        
        # Detect features in the crop
        kp_crop, des_crop = self.orb.detectAndCompute(crop, None)
        
        if des_crop is None or len(kp_crop) < MIN_FEATURES:
            print(f"Too few features detected: {0 if des_crop is None else len(kp_crop)}")
            return False
        
        print(f"\n{'='*60}")
        print(f"Creating reference from frame {self.frame_count}")
        print(f"Detected {len(kp_crop)} features in card_game")
        
        # Build 3D plane coordinates for each keypoint
        # The crop represents the object lying flat (Z=0)
        crop_h, crop_w = crop.shape[:2]
        
        # Scale factors: pixels to meters
        sx = self.obj_width / crop_w
        sy = self.obj_height / crop_h
        
        # Convert keypoint positions to 3D world coordinates
        plane_ref = []
        valid_kp = []
        valid_des = []
        
        for i, kp in enumerate(kp_crop):
            u, v = kp.pt
            
            # Convert pixel position to world coordinates
            # Origin at top-left of crop, Y points down in image
            X = u * sx
            Y = -v * sy  # Negative because image Y is downward
            Z = 0.0  # Flat plane assumption
            
            plane_ref.append([X, Y, Z])
            
            # Adjust keypoint position to full frame coordinates
            kp_adjusted = cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1,
                                       kp.size, kp.angle, kp.response,
                                       kp.octave, kp.class_id)
            valid_kp.append(kp_adjusted)
            valid_des.append(des_crop[i])
        
        # Store reference
        self.ref_image = frame.copy()
        self.ref_kp = valid_kp
        self.ref_des = np.array(valid_des)
        self.plane_ref = np.array(plane_ref, dtype=np.float32)
        self.ref_bbox = bbox
        self.has_reference = True
        self.reference_capture_frame = self.frame_count
        
        print(f"Reference created with {len(self.ref_kp)} features")
        print(f"Object dimensions: {self.obj_width*1000:.1f}mm x {self.obj_height*1000:.1f}mm")
        print(f"{'='*60}\n")
        
        return True
    
    def track_pose(self, frame, bbox):
        """Track 6DOF pose using feature matching and PnP."""
        x1, y1, x2, y2 = bbox
        
        # Extract features from current frame (full frame, not just bbox)
        kp_live, des_live = self.orb.detectAndCompute(frame, None)
        
        if des_live is None or len(kp_live) < MIN_FEATURES:
            return None, "No features in frame"
        
        # Match features
        matches = self.bf.knnMatch(self.ref_des, des_live, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < MIN_MATCH_COUNT:
            return None, f"Few matches ({len(good_matches)})"
        
        # Build 3D-2D correspondences
        obj_pts = []
        img_pts = []
        
        for m in good_matches:
            # 3D point from reference
            P3 = self.plane_ref[m.queryIdx]
            if np.any(np.isnan(P3)):
                continue
            
            # 2D point in current frame
            pt2d = kp_live[m.trainIdx].pt
            
            obj_pts.append(P3)
            img_pts.append(pt2d)
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        
        if len(obj_pts) < MIN_PNP_POINTS:
            return None, f"Few PnP points ({len(obj_pts)})"
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, self.dist,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            return None, "PnP failed"
        
        inliers = inliers.reshape(-1)
        obj_in = obj_pts[inliers]
        img_in = img_pts[inliers]
        
        # Compute reprojection error
        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        errors = np.linalg.norm(proj - img_in, axis=1)
        mean_err = float(np.mean(errors))
        
        # Refine with inliers only
        if mean_err < REPROJ_ERROR_THRESH * 2:
            keep_idx = np.where(errors < REPROJ_ERROR_THRESH)[0]
            if len(keep_idx) >= MIN_PNP_POINTS:
                obj_ref = obj_in[keep_idx]
                img_ref = img_in[keep_idx]
                try:
                    success_ref, rvec_ref, tvec_ref = cv2.solvePnP(
                        obj_ref, img_ref, self.K, self.dist,
                        rvec, tvec, True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success_ref:
                        rvec, tvec = rvec_ref, tvec_ref
                        obj_in, img_in = obj_ref, img_ref
                        
                        # Recalculate error
                        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, self.K, self.dist)
                        proj = proj.reshape(-1, 2)
                        errors = np.linalg.norm(proj - img_in, axis=1)
                        mean_err = float(np.mean(errors))
                except cv2.error:
                    pass
        
        # Store previous pose
        self.prev_rvec = rvec.copy()
        self.prev_tvec = tvec.copy()
        
        # Return pose and inlier info
        pose_data = {
            'rvec': rvec,
            'tvec': tvec,
            'inliers': img_in,
            'n_inliers': len(inliers),
            'n_matches': len(good_matches),
            'reproj_error': mean_err
        }
        
        status = f"Err={mean_err:.2f}px, {len(inliers)} inliers"
        
        return pose_data, status
    
    def draw_visualization(self, frame, bbox, pose_data=None, status_text=""):
        """Draw all visualization elements."""
        display = frame.copy()
        
        if bbox is not None and DRAW_BBOX:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, TARGET_OBJECT, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw pose if available
        if pose_data is not None:
            rvec = pose_data['rvec']
            tvec = pose_data['tvec']
            
            # Draw 3D axes
            axis_len = min(self.obj_width, self.obj_height) * 0.7
            axis_3d = np.float32([
                [0, 0, 0],
                [axis_len, 0, 0],
                [0, axis_len, 0],
                [0, 0, axis_len]
            ])
            
            imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, self.K, self.dist)
            imgpts = imgpts.reshape(-1, 2).astype(int)
            
            origin = tuple(imgpts[0])
            x_end = tuple(imgpts[1])
            y_end = tuple(imgpts[2])
            z_end = tuple(imgpts[3])
            
            # Draw axes: X=Red, Y=Green, Z=Blue
            cv2.line(display, origin, x_end, (0, 0, 255), 3)
            cv2.line(display, origin, y_end, (0, 255, 0), 3)
            cv2.line(display, origin, z_end, (255, 0, 0), 3)
            
            # Draw inliers
            if DRAW_INLIERS and 'inliers' in pose_data:
                for pt in pose_data['inliers']:
                    cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
            
            # Display pose values
            t = tvec.ravel()
            y_offset = 30
            cv2.putText(display, f"X: {t[0]:.3f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"Y: {t[1]:.3f}m", (10, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, f"Z: {t[2]:.3f}m", (10, y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Status text
            cv2.putText(display, status_text, (10, y_offset + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return display
    
    def run(self):
        """Main tracking loop."""
        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not cap.isOpened():
            print(f"Cannot open camera {CAMERA_INDEX}")
            return
        
        print("\n" + "="*60)
        print("Card Game 6DOF Tracker")
        print("="*60)
        print(f"Target object: {TARGET_OBJECT}")
        print(f"Camera index: {CAMERA_INDEX}")
        print("\nWaiting for stable detection to capture reference...")
        print("Press 'q' to quit, 'r' to reset reference\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            self.frame_count += 1
            
            if UNDISTORT and self.dist is not None:
                frame = cv2.undistort(frame, self.K, self.dist)
            
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            # Find target object
            target_det = None
            target_bbox = None
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id]
                    conf = float(box.conf[0])
                    
                    if class_name == TARGET_OBJECT and conf >= MIN_DETECTION_CONFIDENCE:
                        xyxy = box.xyxy[0].cpu().numpy()
                        target_bbox = [int(xyxy[0]), int(xyxy[1]), 
                                      int(xyxy[2]), int(xyxy[3])]
                        target_det = (class_name, conf)
                        break
                
                if target_det:
                    break
            
            # State machine
            if target_det is None:
                # No detection
                self.stable_bbox_buffer.clear()
                display = self.draw_visualization(frame, None)
                cv2.putText(display, "Waiting for card_game detection...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            elif not self.has_reference:
                # Detection found, but no reference yet
                class_name, conf = target_det
                
                # Check if bbox is stable
                if self.is_bbox_stable(target_bbox):
                    # Try to create reference
                    success = self.create_reference(frame, target_bbox)
                    
                    if not success:
                        self.stable_bbox_buffer.clear()
                        display = self.draw_visualization(frame, target_bbox)
                        cv2.putText(display, "Poor quality, repositioning...", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        display = self.draw_visualization(frame, target_bbox)
                        cv2.putText(display, "Reference captured! Tracking...", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Still stabilizing
                    stable_count = len(self.stable_bbox_buffer)
                    display = self.draw_visualization(frame, target_bbox)
                    cv2.putText(display, 
                               f"Hold steady... {stable_count}/{STABLE_FRAMES_NEEDED}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            else:
                # Reference exists, track pose
                pose_data, status = self.track_pose(frame, target_bbox)
                
                if pose_data is None:
                    # Tracking failed
                    display = self.draw_visualization(frame, target_bbox)
                    cv2.putText(display, f"Tracking lost: {status}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # Successful tracking
                    display = self.draw_visualization(frame, target_bbox, pose_data, status)
            
            # Show frame
            cv2.imshow("Card Game 6DOF Tracker", display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset reference
                print("\nResetting reference...")
                self.has_reference = False
                self.ref_image = None
                self.ref_kp = None
                self.ref_des = None
                self.plane_ref = None
                self.stable_bbox_buffer.clear()
                self.prev_tvec = None
                self.prev_rvec = None
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nTracking stopped.")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    tracker = CardGameTracker()
    tracker.run()