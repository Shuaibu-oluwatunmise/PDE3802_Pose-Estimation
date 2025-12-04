"""
Complete Pose Estimation System with Reference Loading - RASPBERRY PI VERSION
- Loads pre-captured references at startup
- State machine: UNINITIALIZED → TRACKING → LOST → RECOVERING
- YOLO always runs (for visualization and recovery)
- Homography independent after initialization
- Automatic recovery when tracking is lost
- 🟢 RASPBERRY PI CSI CAMERA via GStreamer/libcamera
"""

import os
# RPI CSI: Set environment variables before importing GI
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
import json
from glob import glob
from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ==============================================================================
# TRACKING STATES
# ==============================================================================
class TrackingState:
    UNINITIALIZED = 0  # Never found reference
    TRACKING = 1       # Actively tracking
    LOST = 2          # Was tracking, now lost
    
    @staticmethod
    def name(state):
        names = {0: "UNINITIALIZED", 1: "TRACKING", 2: "LOST"}
        return names.get(state, "UNKNOWN")

# ==============================================================================
# ARUCO CONFIG
# ==============================================================================
ARUCO_DICT = cv2.aruco.DICT_4X4_50

ARUCO_IDS = {
    "wallet": 0,
    "headset": 1,
}

ARUCO_MARKER_SIZES = {
    "wallet": 50.0,   # mm
    "headset": 32.0,  # mm
}

# ==============================================================================
# OBJECT CONFIGS
# ==============================================================================
OBJECT_CONFIGS = {
    # Homography objects - will load from references
    "repair_mat": {
        "label_substring": "repair_mat",
        "axis_color": (0, 255, 0),       # Green
        "method": "homography",
        "min_matches": 15,               # For tracking
        "min_inlier_ratio": 0.3,
        "init_min_matches": 20,          # Higher threshold for initialization
        "init_min_inlier_ratio": 0.4,
        "recovery_min_matches": 15,      # Lower threshold for recovery
        "recovery_min_inlier_ratio": 0.3,
        "max_failures_before_lost": 5,
        "ema_alpha": 0.2,
    },
    "game_box": {
        "label_substring": "game_box",
        "axis_color": (255, 0, 0),       # Blue
        "method": "homography",
        "min_matches": 15,
        "min_inlier_ratio": 0.4,
        "init_min_matches": 20,
        "init_min_inlier_ratio": 0.5,
        "recovery_min_matches": 15,
        "recovery_min_inlier_ratio": 0.35,
        "max_failures_before_lost": 5,
        "ema_alpha": 0.2,
    },
    "notebook": {
        "label_substring": "notebook",
        "axis_color": (0, 0, 255),       # Red
        "method": "homography",
        "min_matches": 15,
        "min_inlier_ratio": 0.4,
        "init_min_matches": 20,
        "init_min_inlier_ratio": 0.5,
        "recovery_min_matches": 15,
        "recovery_min_inlier_ratio": 0.35,
        "max_failures_before_lost": 5,
        "ema_alpha": 0.2,
    },
    # ArUco objects
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


def load_reference(reference_dir, object_name):
    """Load a pre-saved reference image"""
    json_files = glob(os.path.join(reference_dir, "*.json"))
    
    if not json_files:
        return None
    
    # Load all metadata
    references = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            metadata['_json_path'] = json_file
            metadata['_base_path'] = json_file.replace('.json', '')
            references.append(metadata)
    
    # Find matching object
    matching = [r for r in references if r['object_name'].lower() == object_name.lower()]
    
    if not matching:
        return None
    
    # Use most recent
    matching.sort(key=lambda x: x['timestamp'], reverse=True)
    selected = matching[0]
    
    # Load image
    image_path = os.path.join(reference_dir, selected['image_file'])
    image = cv2.imread(image_path)
    
    if image is None:
        return None
    
    # Load features
    features_path = selected['_base_path'] + '_features.npz'
    
    if not os.path.exists(features_path):
        return None
    
    features_data = np.load(features_path)
    kp_data = features_data['keypoints']
    descriptors = features_data['descriptors']
    
    # Reconstruct keypoints
    keypoints = []
    for kp_info in kp_data:
        kp = cv2.KeyPoint(
            x=float(kp_info[0]),
            y=float(kp_info[1]),
            size=float(kp_info[2]),
            angle=float(kp_info[3]),
            response=float(kp_info[4]),
            octave=int(kp_info[5]),
            class_id=int(kp_info[6])
        )
        keypoints.append(kp)
    
    return {
        'image': image,
        'keypoints': keypoints,
        'descriptors': descriptors,
        'metadata': selected,
        'object_name': selected['object_name'],
        'width_mm': selected['dimensions_mm']['width'],
        'height_mm': selected['dimensions_mm']['height']
    }


class ReferencePoseEstimator:
    def __init__(self, yolo_model_path, reference_dir="reference_images"):
        # 🟢 RPI CSI: Pipeline instead of VideoCapture
        self.pipeline = None
        self.sink = None
        self.reference_dir = reference_dir

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

        # Frame skipping
        self.frame_idx = 0
        self.yolo_every_n = 2  # Run YOLO every 2 frames
        
        self.last_bboxes = {name: None for name in OBJECT_CONFIGS}
        self.last_confidences = {name: 0.0 for name in OBJECT_CONFIGS}

        # Camera parameters
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DIST_COEFFS
        print("✓ Loaded camera parameters from camera_params.py")

        # ORB for homography objects
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        print(f"✓ ORB initialized (2000 features)")

        # Build per-object state
        self.targets = {}
        self._initialize_targets()

    def _initialize_targets(self):
        """Initialize target states and load references"""
        for obj_name, cfg in OBJECT_CONFIGS.items():
            label_sub = cfg["label_substring"].lower()
            class_id = next(
                (cid for cid, cname in self.class_names.items()
                 if label_sub in cname.lower()),
                None,
            )
            
            if class_id is None:
                print(f"WARNING: '{obj_name}' class not found in model!")
                continue
            
            print(f"✓ {obj_name.capitalize()} class found: ID={class_id} ({cfg['method']})")

            state = {
                "class_id": class_id,
                "method": cfg["method"],
                "axis_color": cfg["axis_color"],
                "ema_alpha": cfg.get("ema_alpha", 0.2),
                "rvec_smooth": None,
                "tvec_smooth": None,
            }

            if cfg["method"] == "homography":
                # Try to load reference
                ref = load_reference(self.reference_dir, obj_name)
                
                if ref is not None:
                    kp_count = len(ref['keypoints'])
                    print(f"  → Loaded reference: {kp_count} features, {ref['width_mm']}x{ref['height_mm']}mm")
                    
                    # Store reference data
                    state["ref_image"] = cv2.cvtColor(ref['image'], cv2.COLOR_BGR2GRAY)
                    state["ref_keypoints"] = ref['keypoints']
                    state["ref_descriptors"] = ref['descriptors']
                    state["width_mm"] = ref['width_mm']
                    state["height_mm"] = ref['height_mm']
                    
                    # Setup 3D points
                    half_w = ref['width_mm'] / 2.0
                    half_h = ref['height_mm'] / 2.0
                    state["object_points_3d"] = np.array([
                        [-half_w,  half_h, 0],
                        [ half_w,  half_h, 0],
                        [ half_w, -half_h, 0],
                        [-half_w, -half_h, 0],
                    ], dtype=np.float32)
                    
                    # Tracking state
                    state["tracking_state"] = TrackingState.UNINITIALIZED
                    state["tracking_failures"] = 0
                    state["max_failures"] = cfg.get("max_failures_before_lost", 5)
                    
                    # Thresholds
                    state["min_matches"] = cfg.get("min_matches", 15)
                    state["min_inlier_ratio"] = cfg.get("min_inlier_ratio", 0.3)
                    state["init_min_matches"] = cfg.get("init_min_matches", 20)
                    state["init_min_inlier_ratio"] = cfg.get("init_min_inlier_ratio", 0.4)
                    state["recovery_min_matches"] = cfg.get("recovery_min_matches", 15)
                    state["recovery_min_inlier_ratio"] = cfg.get("recovery_min_inlier_ratio", 0.3)
                    
                    state["reference_loaded"] = True
                else:
                    print(f"  ✗ No reference found for {obj_name} in {self.reference_dir}")
                    state["reference_loaded"] = False

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

    # 🟢 RPI CSI: GStreamer Pipeline Setup
    def start_camera(self):
        """Start Raspberry Pi CSI Camera via GStreamer/libcamera"""
        print(f"Starting Raspberry Pi CSI Camera via GStreamer/libcamera...")
        
        # Init GStreamer
        Gst.init(None)

        # Pipeline string matching Resolution in camera_params.py (640x480)
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

    # 🟢 RPI CSI: Frame Pulling Logic
    def pull_frame(self, timeout_ns=10_000_000):
        """Grab one BGR frame from appsink, or None if timed out."""
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
            # Convert raw BGR data to NumPy array
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3)
            return frame.copy() # Return copy to ensure data persists
        finally:
            buf.unmap(mapinfo)

    def detect_objects(self, frame):
        """Run YOLO detection"""
        results = self.yolo_model(frame, verbose=False, imgsz=self.yolo_imgsz, conf=0.3, iou=0.5)
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

    # ========== HOMOGRAPHY METHODS ==========
    
    def try_match_reference(self, frame_region, ref_descriptors, ref_keypoints, 
                           min_matches, min_inlier_ratio):
        """
        Try to match reference in a frame region
        Returns (success, H, inlier_ratio, good_matches_count)
        """
        gray = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY) if len(frame_region.shape) == 3 else frame_region
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 10:
            return False, None, 0, 0
        
        # Match
        matches = self.bf_matcher.knnMatch(ref_descriptors, descriptors, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < min_matches:
            return False, None, 0, len(good_matches)
        
        # Homography
        src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        
        if H is None or mask is None:
            return False, None, 0, len(good_matches)
        
        inliers = mask.ravel().tolist()
        inlier_ratio = sum(inliers) / len(inliers)
        
        if inlier_ratio < min_inlier_ratio:
            return False, None, inlier_ratio, len(good_matches)
        
        return True, H, inlier_ratio, len(good_matches)
    
    def track_homography_full_frame(self, frame, obj_name):
        """Track object in full frame using reference"""
        state = self.targets[obj_name]
        
        success, H, inlier_ratio, good_matches = self.try_match_reference(
            frame,
            state["ref_descriptors"],
            state["ref_keypoints"],
            state["min_matches"],
            state["min_inlier_ratio"]
        )
        
        if not success:
            return None, inlier_ratio, good_matches
        
        # Get corners
        h_ref, w_ref = state["ref_image"].shape
        ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)
        corners_frame = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)
        
        return corners_frame, inlier_ratio, good_matches

    def detect_aruco_in_roi(self, frame, bbox):
        """Detect ArUco in ROI"""
        x1, y1, x2, y2 = bbox
        
        pad = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_aruco_api:
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray_roi)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray_roi, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is None or len(ids) == 0:
            return None, None
        
        corners_frame = []
        for corner_set in corners:
            corner_adjusted = corner_set.copy()
            corner_adjusted[0, :, 0] += x1
            corner_adjusted[0, :, 1] += y1
            corners_frame.append(corner_adjusted)
        
        return corners_frame, ids.flatten()

    # ========== POSE ESTIMATION ==========
    
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

    # ========== DRAWING ==========
    
    def draw_detection_box(self, frame, obj_name, bbox, state_info=""):
        """Draw YOLO detection box"""
        x1, y1, x2, y2 = bbox
        state = self.targets[obj_name]
        
        # Color based on tracking state
        if state["method"] == "homography":
            tracking_state = state.get("tracking_state", TrackingState.UNINITIALIZED)
            if tracking_state == TrackingState.TRACKING:
                color = (0, 255, 0)  # Green
            elif tracking_state == TrackingState.LOST:
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{obj_name}"
        if state_info:
            label += f" [{state_info}]"
        
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_tracked_corners(self, frame, obj_name, corners):
        """Draw tracked corners"""
        corners_int = corners.astype(int)
        color = self.targets[obj_name]["axis_color"]
        cv2.polylines(frame, [corners_int], True, color, 3)
        return frame

    def draw_3d_axes(self, frame, obj_name, rvec, tvec, length=50):
        """Draw 3D axes at pose origin (for ArUco)"""
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

    def draw_3d_axes_at_bbox_center(self, frame, obj_name, bbox, rvec, tvec, length=50):
        """Draw 3D axes at bbox center with correct orientation (for Homography)"""
        color = self.targets[obj_name]["axis_color"]
        
        # Get bbox center
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Define axes in 3D space
        axis_points_3d = np.float32([
            [0, 0, 0],       # Origin
            [length, 0, 0],  # X
            [0, length, 0],  # Y
            [0, 0, length]   # Z
        ])
        
        # Project to 2D using the estimated pose
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, 
            self.camera_matrix, self.dist_coeffs
        )
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        
        # Get the projected origin
        projected_origin = axis_points_2d[0]
        
        # Calculate offset to move origin to bbox center
        offset_x = center_x - projected_origin[0]
        offset_y = center_y - projected_origin[1]
        
        # Apply offset to all points
        axis_points_2d[:, 0] += offset_x
        axis_points_2d[:, 1] += offset_y
        
        # Draw axes at new position (bbox center)
        origin = tuple(axis_points_2d[0])
        
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3, tipLength=0.3)  # X-Red
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3, tipLength=0.3)  # Y-Green
        cv2.arrowedLine(frame, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3, tipLength=0.3)  # Z-Blue
        
        cv2.circle(frame, origin, 8, color, -1)
        
        return frame

    def draw_pose_info(self, frame, obj_name, y_offset_start, rvec_final, tvec_final):
        """Draw pose info"""
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
        """Main loop"""
        print("\n" + "="*60)
        print("REFERENCE-BASED POSE ESTIMATION SYSTEM - RASPBERRY PI")
        print("="*60)
        print("\nFeatures:")
        print("  • Pre-loaded perfect references")
        print("  • YOLO for detection + recovery")
        print("  • Independent homography tracking")
        print("  • Automatic recovery when lost")
        print("  • 🟢 CSI Camera via GStreamer/libcamera")
        print("\nControls:")
        print("  's' - Save frame")
        print("  ESC - Exit")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return
        
        saved_count = 0
        fps = 0
        frame_count = 0
        last_time = time.time()
        
        try:
            while True:
                # 🟢 RPI CSI: Pull frame instead of cap.read()
                frame = self.pull_frame()
                if frame is None:
                    # Timeout or no data; let GUI update
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
                
                # YOLO detection (always runs)
                self.frame_idx += 1
                run_yolo = (self.frame_idx % self.yolo_every_n == 0)
                
                if run_yolo:
                    bboxes, confidences = self.detect_objects(frame)
                    self.last_bboxes = bboxes
                    self.last_confidences = confidences
                else:
                    bboxes = self.last_bboxes
                    confidences = self.last_confidences
                
                y_info_offset = 30
                tracked_count = 0
                
                # Process each object
                for obj_name, bbox in bboxes.items():
                    state = self.targets.get(obj_name)
                    if state is None:
                        continue
                    
                    # Draw YOLO detection box (always)
                    if bbox is not None:
                        state_name = TrackingState.name(state.get("tracking_state", TrackingState.UNINITIALIZED)) if state["method"] == "homography" else "ARUCO"
                        display_frame = self.draw_detection_box(display_frame, obj_name, bbox, state_name)
                    
                    # === HOMOGRAPHY PATH ===
                    if state["method"] == "homography":
                        if not state.get("reference_loaded"):
                            continue
                        
                        tracking_state = state["tracking_state"]
                        
                        # STATE MACHINE
                        if tracking_state == TrackingState.UNINITIALIZED:
                            # Try to initialize using YOLO bbox
                            if bbox is not None and confidences[obj_name] > 0.7:
                                roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                success, H, inlier_ratio, good_matches = self.try_match_reference(
                                    roi,
                                    state["ref_descriptors"],
                                    state["ref_keypoints"],
                                    state["init_min_matches"],
                                    state["init_min_inlier_ratio"]
                                )
                                
                                if success:
                                    print(f"✓ {obj_name} INITIALIZED! (matches={good_matches}, inlier_ratio={inlier_ratio:.2f})")
                                    state["tracking_state"] = TrackingState.TRACKING
                                    state["tracking_failures"] = 0
                        
                        elif tracking_state == TrackingState.TRACKING:
                            # Pure homography tracking (full frame)
                            corners, inlier_ratio, good_matches = self.track_homography_full_frame(frame, obj_name)
                            
                            if corners is not None:
                                # Success - reset failure counter
                                state["tracking_failures"] = 0
                                
                                # Estimate pose
                                rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                                
                                if rvec_raw is not None:
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
                                    
                                    # Draw axes at bbox center (corners hidden for clean visualization)
                                    # Note: bbox might be None if YOLO skipped, use last known or corners
                                    if bbox is not None:
                                        display_frame = self.draw_3d_axes_at_bbox_center(display_frame, obj_name, bbox, rvec_final, tvec_final)
                                    
                                    y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, rvec_final, tvec_final)
                                    tracked_count += 1
                            else:
                                # Tracking failed
                                state["tracking_failures"] += 1
                                
                                if state["tracking_failures"] >= state["max_failures"]:
                                    print(f"⚠ {obj_name} LOST (failures={state['tracking_failures']})")
                                    state["tracking_state"] = TrackingState.LOST
                        
                        elif tracking_state == TrackingState.LOST:
                            # Recovery mode - use YOLO bbox to re-find reference
                            if bbox is not None:
                                roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                success, H, inlier_ratio, good_matches = self.try_match_reference(
                                    roi,
                                    state["ref_descriptors"],
                                    state["ref_keypoints"],
                                    state["recovery_min_matches"],
                                    state["recovery_min_inlier_ratio"]
                                )
                                
                                if success:
                                    print(f"✓ {obj_name} RECOVERED! (matches={good_matches}, inlier_ratio={inlier_ratio:.2f})")
                                    state["tracking_state"] = TrackingState.TRACKING
                                    state["tracking_failures"] = 0
                    
                    # === ARUCO PATH ===
                    elif state["method"] == "aruco":
                        if bbox is None:
                            continue
                        
                        aruco_corners, aruco_ids = self.detect_aruco_in_roi(frame, bbox)
                        if aruco_corners is None or aruco_ids is None:
                            continue
                        
                        target_id = state["aruco_id"]
                        if target_id not in aruco_ids:
                            continue
                        
                        idx = np.where(aruco_ids == target_id)[0][0]
                        corners = aruco_corners[idx][0]
                        
                        rvec_raw, tvec_raw = self.estimate_pose_pnp(obj_name, corners)
                        if rvec_raw is None:
                            continue
                        
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
                        
                        # Draw ArUco corners and axes at marker position (traditional way)
                        display_frame = self.draw_tracked_corners(display_frame, obj_name, corners)
                        display_frame = self.draw_3d_axes(display_frame, obj_name, rvec_final, tvec_final)
                        y_info_offset = self.draw_pose_info(display_frame, obj_name, y_info_offset, rvec_final, tvec_final)
                        tracked_count += 1
                
                # UI
                cv2.putText(display_frame, f"FPS: {fps}", (display_frame.shape[1]-100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if tracked_count > 0:
                    cv2.putText(display_frame, f"TRACKING {tracked_count}/5 OBJECTS", 
                                (10, display_frame.shape[0]-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('RPI Pose Estimation', display_frame)
                
                # Keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:
                    print("\nExiting...")
                    break
                
                elif key == ord('s'):
                    filename = f"rpi_tracking_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display_frame)
                    saved_count += 1
                    print(f"✓ Saved: {filename}")
        
        finally:
            # 🟢 RPI CSI: Null state for pipeline
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            cv2.destroyAllWindows()


def main():
    print("\n" + "="*60)
    print("SETUP - RASPBERRY PI VERSION")
    print("="*60)
    
    yolo_model_path = 'runs/detect/yolov8n_detect_V1/weights/best.pt'
    reference_dir = 'reference_images'
    
    if not os.path.exists(reference_dir):
        print(f"✗ Reference directory not found: {reference_dir}")
        print(f"Run capture_references.py first!")
        return
    
    # 🟢 RPI CSI: No camera_index needed
    estimator = ReferencePoseEstimator(yolo_model_path, reference_dir)
    
    print("\n--- Loaded Objects ---")
    for obj_name, state in estimator.targets.items():
        if state["method"] == "homography":
            if state.get("reference_loaded"):
                kp_count = len(state['ref_keypoints'])
                print(f"✓ {obj_name}: Reference loaded ({kp_count} features)")
            else:
                print(f"✗ {obj_name}: No reference")
        else:
            print(f"✓ {obj_name}: ArUco (ID {state['aruco_id']})")
    print("="*60 + "\n")

    estimator.run()


if __name__ == "__main__":
    main()