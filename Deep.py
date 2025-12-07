#!/usr/bin/env python3
"""
Optimized 5-Object 6DOF Pose Tracker for Raspberry Pi 4
- Reduced computation, memory usage, and latency
- Frame skipping, smaller resolutions, selective processing
- Maintains accuracy while improving FPS
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

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations

from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

# ============================================================================
# OPTIMIZED CONFIGURATION FOR RPI 4
# ============================================================================
YOLO_MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'

# CAMERA SETTINGS - REDUCE RESOLUTION
CAMERA_WIDTH = 320  # Reduced from 640
CAMERA_HEIGHT = 240  # Reduced from 480
CAMERA_FPS = 20  # Reduced from 30

# PERFORMANCE OPTIMIZATION FLAGS
SKIP_FRAMES = 1  # Process every Nth frame (1 = every frame, 2 = every other frame)
YOLO_INTERVAL = 3  # Run YOLO every N processed frames
TRACK_ONLY_DETECTED = True  # Only track objects detected by YOLO in current/last frame

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

# DETECTION THRESHOLDS
MIN_DETECTION_CONFIDENCE = 0.65  # Slightly lower for more detections
STABLE_FRAMES_NEEDED = 4  # Reduced from 5

# FEATURE TRACKING - REDUCED PARAMETERS
ORB_FEATURES = 500  # Reduced from 2000
MIN_FEATURES = 10  # Reduced from 20
MIN_MATCH_COUNT = 8  # Reduced from 15
MIN_PNP_POINTS = 6  # Reduced from 8
MIN_PNP_INLIERS = 6  # Reduced from 8
REPROJ_ERROR_THRESH = 5.0  # Increased from 3.0 (more tolerant)

# VISUALIZATION - TURN OFF FOR MAX PERFORMANCE
DRAW_BBOX = True
DRAW_INLIERS = False  # Disabled for performance
DRAW_AXES = True
UNDISTORT = False  # Keep disabled on RPi

CAMERA_FRAME = "camera_link_G4"

# ============================================================================
# OPTIMIZED UTILITY FUNCTIONS
# ============================================================================

def load_calibration():
    try:
        K = np.array(CAMERA_MATRIX, dtype=np.float32)
        # Scale camera matrix for smaller resolution
        scale_x = CAMERA_WIDTH / 640
        scale_y = CAMERA_HEIGHT / 480
        K[0, 0] *= scale_x  # fx
        K[0, 2] *= scale_x  # cx
        K[1, 1] *= scale_y  # fy
        K[1, 2] *= scale_y  # cy
        
        dist = np.array(DIST_COEFFS, dtype=np.float32) if DIST_COEFFS is not None else None
        print(f"Camera calibration loaded! Scaled for {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        return K, dist
    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")
        K = np.array([[400, 0, 160], [0, 400, 120], [0, 0, 1]], dtype=np.float32)
        return K, None

def bbox_iou(box1, box2):
    """Optimized IoU calculation"""
    x1_tl, y1_tl, x1_br, y1_br = box1
    x2_tl, y2_tl, x2_br, y2_br = box2
    
    xi1 = max(x1_tl, x2_tl)
    yi1 = max(y1_tl, y2_tl)
    xi2 = min(x1_br, x2_br)
    yi2 = min(y1_br, y2_br)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = (x1_br - x1_tl) * (y1_br - y1_tl)
    area2 = (x2_br - x2_tl) * (y2_br - y2_tl)
    
    return intersection / (area1 + area2 - intersection)

# ============================================================================
# OPTIMIZED FEATURE TRACKER
# ============================================================================

class OptimizedFeatureTracker:
    """Optimized tracker for RPi 4"""
    def __init__(self, obj_name, obj_specs, K, dist):
        self.obj_name = obj_name
        self.obj_width = obj_specs["width_m"]
        self.obj_height = obj_specs["height_m"]
        self.color = obj_specs["color"]
        
        # Create separate ORB for reference (higher quality) and tracking (faster)
        self.orb_ref = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=7)
        self.orb_track = cv2.ORB_create(nfeatures=ORB_FEATURES//2, fastThreshold=20)  # Faster
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.K = K
        self.dist = dist
        
        self.ref_image = None
        self.ref_kp = None
        self.ref_des = None
        self.plane_ref = None
        self.has_reference = False
        self.stable_bbox_buffer = []
        self.last_detected_frame = 0
        self.consecutive_failures = 0
        self.max_failures = 10  # Reset after too many failures
    
    def is_bbox_stable(self, bbox, frame_id):
        self.stable_bbox_buffer.append((frame_id, bbox))
        # Keep only recent frames
        self.stable_bbox_buffer = [(fid, b) for fid, b in self.stable_bbox_buffer if frame_id - fid < 30]
        
        if len(self.stable_bbox_buffer) < STABLE_FRAMES_NEEDED:
            return False
        
        # Check if recent boxes are stable
        recent_boxes = [b for fid, b in self.stable_bbox_buffer[-STABLE_FRAMES_NEEDED:]]
        first_box = recent_boxes[0]
        for box in recent_boxes[1:]:
            if bbox_iou(first_box, box) < 0.8:  # Slightly more tolerant
                return False
        return True
    
    def create_reference(self, frame, bbox, frame_id, logger):
        x1, y1, x2, y2 = bbox
        # Use small margin
        m = 5
        x1, y1 = max(0, x1-m), max(0, y1-m)
        x2, y2 = min(frame.shape[1], x2+m), min(frame.shape[0], y2+m)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        
        # Resize crop for faster processing (if large)
        h, w = crop.shape[:2]
        if w > 200 or h > 200:
            crop = cv2.resize(crop, (w//2, h//2))
            scale_factor = 2.0
        else:
            scale_factor = 1.0
        
        kp_crop, des_crop = self.orb_ref.detectAndCompute(crop, None)
        if des_crop is None or len(kp_crop) < MIN_FEATURES:
            return False
        
        # Scale keypoints back to original crop size
        for kp in kp_crop:
            kp.pt = (kp.pt[0] * scale_factor, kp.pt[1] * scale_factor)
            kp.size *= scale_factor
        
        crop_h, crop_w = crop.shape[:2] * scale_factor
        sx = self.obj_width / crop_w
        sy = self.obj_height / crop_h
        
        plane_ref = []
        valid_kp = []
        valid_des = []
        
        center_x = self.obj_width / 2.0
        center_y = self.obj_height / 2.0
        
        for i, kp in enumerate(kp_crop):
            u, v = kp.pt
            
            X = (u * sx) - center_x
            Y = (-v * sy) + center_y
            Z = 0.0
            
            plane_ref.append([X, Y, Z])
            kp_adjusted = cv2.KeyPoint(kp.pt[0] + x1, kp.pt[1] + y1,
                                       kp.size, kp.angle, kp.response,
                                       kp.octave, kp.class_id)
            valid_kp.append(kp_adjusted)
            valid_des.append(des_crop[i])
        
        self.ref_image = frame.copy()
        self.ref_kp = valid_kp
        self.ref_des = np.array(valid_des)
        self.plane_ref = np.array(plane_ref, dtype=np.float32)
        self.has_reference = True
        self.last_detected_frame = frame_id
        self.consecutive_failures = 0
        
        logger.info(f"✓ {self.obj_name}: Reference created ({len(self.ref_kp)} features)")
        return True
    
    def track_pose(self, frame, frame_id):
        if not self.has_reference or self.consecutive_failures > self.max_failures:
            return None, "No reference or too many failures"
        
        # Skip tracking if object hasn't been seen recently
        if TRACK_ONLY_DETECTED and (frame_id - self.last_detected_frame) > 10:
            return None, "Not recently detected"
        
        # Use faster ORB for tracking
        kp_live, des_live = self.orb_track.detectAndCompute(frame, None)
        if des_live is None or len(kp_live) < MIN_FEATURES:
            self.consecutive_failures += 1
            return None, "No features"
        
        # Limit matches for speed
        matches = self.bf.knnMatch(self.ref_des, des_live, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:  # More tolerant
                    good_matches.append(m)
        
        if len(good_matches) < MIN_MATCH_COUNT:
            self.consecutive_failures += 1
            return None, f"Few matches ({len(good_matches)})"
        
        # Use first N matches for speed
        if len(good_matches) > 50:
            good_matches = good_matches[:50]
        
        obj_pts = []
        img_pts = []
        for m in good_matches:
            P3 = self.plane_ref[m.queryIdx]
            pt2d = kp_live[m.trainIdx].pt
            obj_pts.append(P3)
            img_pts.append(pt2d)
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        
        if len(obj_pts) < MIN_PNP_POINTS:
            self.consecutive_failures += 1
            return None, f"Few PnP points ({len(obj_pts)})"
        
        # Use solvePnP instead of RANSAC for speed (if previous pose available)
        try:
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.K, self.dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
                useExtrinsicGuess=False
            )
        except:
            self.consecutive_failures += 1
            return None, "PnP failed"
        
        if not success:
            self.consecutive_failures += 1
            return None, "PnP failed"
        
        self.consecutive_failures = 0
        self.last_detected_frame = frame_id
        
        pose_data = {
            'rvec': rvec, 'tvec': tvec,
            'n_matches': len(good_matches)
        }
        status = f"{len(good_matches)} matches"
        return pose_data, status
    
    def reset(self):
        self.has_reference = False
        self.ref_image = None
        self.ref_kp = None
        self.ref_des = None
        self.plane_ref = None
        self.stable_bbox_buffer.clear()
        self.consecutive_failures = 0

# ============================================================================
# OPTIMIZED MAIN TRACKER
# ============================================================================

class Optimized5ObjectTracker(Node):
    def __init__(self):
        super().__init__('optimized_5_object_tracker_node')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pipeline = None
        self.sink = None
        
        self.get_logger().info("Loading optimized YOLO model...")
        # Use smaller YOLO model if available
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.get_logger().info("✓ YOLO model loaded")
        
        self.K, self.dist = load_calibration()
        
        # Initialize ArUco detector once
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Faster
        
        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_aruco_api = True
        except AttributeError:
            self.use_new_aruco_api = False
        
        # Create optimized feature trackers
        self.feature_trackers = {}
        for obj_name, obj_specs in OBJECT_SPECS.items():
            if obj_specs["method"] == "feature":
                self.feature_trackers[obj_name] = OptimizedFeatureTracker(
                    obj_name, obj_specs, self.K, self.dist
                )
        
        # Performance tracking
        self.frame_count = 0
        self.processed_frames = 0
        self.last_yolo_frame = 0
        self.yolo_detections_cache = {}
        self.last_aruco_frame = 0
        self.aruco_cache = {}
        
        self.get_logger().info(f"Optimized tracker initialized for RPi 4")
        self.get_logger().info(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}fps")
        self.get_logger().info(f"Skip frames: {SKIP_FRAMES}, YOLO interval: {YOLO_INTERVAL}")
    
    def publish_tf(self, obj_name, rvec, tvec):
        try:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = CAMERA_FRAME
            t.child_frame_id = f"{obj_name}_frame"
            t.transform.translation.x = float(tvec[0, 0])
            t.transform.translation.y = float(tvec[1, 0])
            t.transform.translation.z = float(tvec[2, 0])
            
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            quaternion = tf_transformations.quaternion_from_matrix(
                np.vstack([np.hstack([rotation_matrix, [[0], [0], [0]]]), [0, 0, 0, 1]])
            )
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]
            
            self.tf_broadcaster.sendTransform(t)
        except Exception as e:
            self.get_logger().error(f"TF publish error for {obj_name}: {e}")
    
    def start_camera(self):
        self.get_logger().info(f"Starting RPi CSI Camera at {CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS}fps...")
        Gst.init(None)
        gst_str = (
            f"libcamerasrc ! "
            f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},format=NV12,framerate={CAMERA_FPS}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true"  # Reduced buffer
        )
        try:
            self.pipeline = Gst.parse_launch(gst_str)
            self.sink = self.pipeline.get_by_name("sink")
            self.pipeline.set_state(Gst.State.PLAYING)
            self.get_logger().info("✓ Camera started")
            return True
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")
            return False
    
    def pull_frame(self, timeout_ns=5_000_000):  # Shorter timeout
        if self.sink is None:
            return None
        
        # Skip frames if needed
        if self.frame_count % (SKIP_FRAMES + 1) != 0:
            self.frame_count += 1
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
            self.frame_count += 1
            self.processed_frames += 1
            return frame.copy()
        finally:
            buf.unmap(mapinfo)
    
    def detect_aruco_markers(self, frame):
        """Optimized ArUco detection with caching"""
        # Only detect ArUco every few frames
        if self.processed_frames - self.last_aruco_frame < 2:  # Run every 2 processed frames
            return self.aruco_cache
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_aruco_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        aruco_detections = {}
        if ids is not None:
            for obj_name, obj_specs in OBJECT_SPECS.items():
                if obj_specs["method"] == "aruco":
                    target_id = obj_specs["aruco_id"]
                    if target_id in ids.flatten():
                        idx = np.where(ids.flatten() == target_id)[0][0]
                        aruco_detections[obj_name] = {
                            'corners': corners[idx][0],
                            'marker_size_m': obj_specs['marker_size_m']
                        }
        
        self.aruco_cache = aruco_detections
        self.last_aruco_frame = self.processed_frames
        return aruco_detections
    
    def run_yolo_detection(self, frame):
        """Run YOLO with reduced frequency"""
        if self.processed_frames - self.last_yolo_frame < YOLO_INTERVAL:
            return self.yolo_detections_cache
        
        # Use half precision if supported
        results = self.yolo_model(frame, verbose=False, half=True if hasattr(self.yolo_model.model, 'half') else False)
        
        detections = {obj_name: None for obj_name in OBJECT_SPECS.keys()}
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                conf = float(box.conf[0])
                
                if class_name in OBJECT_SPECS and conf >= MIN_DETECTION_CONFIDENCE:
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    detections[class_name] = (bbox, conf)
        
        self.yolo_detections_cache = detections
        self.last_yolo_frame = self.processed_frames
        return detections
    
    def run(self):
        if not self.start_camera():
            return
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("OPTIMIZED 5-Object Tracker for RPi 4")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Frame skip: {SKIP_FRAMES}, YOLO every {YOLO_INTERVAL} frames")
        self.get_logger().info("Press 'q' to quit, 'r' to reset\n")
        
        fps_counter = 0
        last_fps_time = self.get_clock().now().seconds_nanoseconds()[0]
        fps = 0.0
        
        while rclpy.ok():
            frame = self.pull_frame()
            if frame is None:
                cv2.waitKey(1)
                continue
            
            # FPS calculation
            fps_counter += 1
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            if current_time - last_fps_time >= 2:  # Update every 2 seconds
                fps = fps_counter / (current_time - last_fps_time)
                self.get_logger().info(f"FPS: {fps:.1f}, Processed: {self.processed_frames}")
                fps_counter = 0
                last_fps_time = current_time
            
            display = frame.copy() if (DRAW_BBOX or DRAW_AXES) else None
            
            # Run YOLO detection (with reduced frequency)
            yolo_detections = self.run_yolo_detection(frame)
            
            # Run ArUco detection
            aruco_detections = self.detect_aruco_markers(frame)
            
            tracking_count = 0
            
            # Process feature-tracked objects
            for obj_name, detection in yolo_detections.items():
                if OBJECT_SPECS[obj_name]["method"] != "feature":
                    continue
                
                tracker = self.feature_trackers[obj_name]
                
                if detection is None:
                    # Clear buffer if not detected
                    if len(tracker.stable_bbox_buffer) > 0:
                        tracker.stable_bbox_buffer.clear()
                    continue
                
                bbox, conf = detection
                
                if display is not None and DRAW_BBOX:
                    color = tracker.color if tracker.has_reference else (128, 128, 128)
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
                    cv2.putText(display, f"{obj_name} ({conf:.1f})", 
                               (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                if not tracker.has_reference:
                    if tracker.is_bbox_stable(bbox, self.processed_frames):
                        tracker.create_reference(frame, bbox, self.processed_frames, self.get_logger())
                else:
                    pose_data, status = tracker.track_pose(frame, self.processed_frames)
                    
                    if pose_data is not None:
                        self.publish_tf(obj_name, pose_data['rvec'], pose_data['tvec'])
                        tracking_count += 1
                        
                        if display is not None and DRAW_AXES:
                            self.draw_axes(display, obj_name, pose_data['rvec'], pose_data['tvec'])
            
            # Process ArUco objects
            for obj_name in ["phone", "estop"]:
                if obj_name in aruco_detections:
                    aruco_data = aruco_detections[obj_name]
                    obj_points = self.get_aruco_points(OBJECT_SPECS[obj_name]['marker_size_m'])
                    img_points = aruco_data['corners'].reshape(-1, 1, 2).astype(np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points, self.K, self.dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    
                    if success:
                        self.publish_tf(obj_name, rvec, tvec)
                        tracking_count += 1
                        
                        if display is not None:
                            if DRAW_BBOX:
                                cv2.aruco.drawDetectedMarkers(display, 
                                                             [aruco_data['corners'].reshape(1, 4, 2)], 
                                                             np.array([[OBJECT_SPECS[obj_name]['aruco_id']]]))
                            if DRAW_AXES:
                                self.draw_axes(display, obj_name, rvec, tvec)
            
            # Display if needed
            if display is not None:
                cv2.putText(display, f"Track: {tracking_count}/5 | FPS: {fps:.1f}", 
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Optimized Tracker (RPi 4)", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.get_logger().info("Resetting all trackers...")
                for tracker in self.feature_trackers.values():
                    tracker.reset()
            
            # Non-blocking ROS spin
            rclpy.spin_once(self, timeout_sec=0.001)
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        self.get_logger().info("Tracker stopped.")
    
    def get_aruco_points(self, marker_size):
        """Pre-computed ArUco points"""
        half = marker_size / 2.0
        return np.array([
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0]
        ], dtype=np.float32)
    
    def draw_axes(self, display, obj_name, rvec, tvec):
        """Draw simple axes"""
        axis_len = 0.05  # Fixed small length for visibility
        axis_3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, self.K, self.dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        
        origin = tuple(imgpts[0])
        cv2.line(display, origin, tuple(imgpts[1]), (0, 0, 255), 2)  # X - red
        cv2.line(display, origin, tuple(imgpts[2]), (0, 255, 0), 2)  # Y - green
        cv2.line(display, origin, tuple(imgpts[3]), (255, 0, 0), 2)  # Z - blue

def main():
    rclpy.init()
    tracker = Optimized5ObjectTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nShutdown")
    finally:
        tracker.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()