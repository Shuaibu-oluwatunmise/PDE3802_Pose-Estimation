#!/usr/bin/env python3
"""
Multi-Object 6DOF Pose Tracker with ROS2 TF Publishing
Tracks card_game, circuit_board, and notebook simultaneously
- Auto-reference capture for each object independently
- Publishes separate TF frames for each
- RPi CSI Camera via GStreamer
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
# CONFIGURATION
# ============================================================================
YOLO_MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'

OBJECT_SPECS = {
    "card_game": {
        "width_m": 0.093,
        "height_m": 0.115,
        "color": (0, 255, 0),  # Green
    },
    "circuit_board": {
        "width_m": 0.210,
        "height_m": 0.210,
        "color": (255, 0, 0),  # Blue
    },
    "notebook": {
        "width_m": 0.147,
        "height_m": 0.209,
        "color": (0, 0, 255),  # Red
    },
}

MIN_DETECTION_CONFIDENCE = 0.70
STABLE_FRAMES_NEEDED = 5
MIN_FEATURES = 20
ORB_FEATURES = 2000
MIN_MATCH_COUNT = 15
MIN_PNP_POINTS = 8
MIN_PNP_INLIERS = 8
REPROJ_ERROR_THRESH = 3.0

DRAW_BBOX = True
DRAW_INLIERS = True
UNDISTORT = False

CAMERA_FRAME = "camera_link_G4"

# ============================================================================
# UTILITY FUNCTIONS
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
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x1_br - x1_tl) * (y1_br - y1_tl)
    area2 = (x2_br - x2_tl) * (y2_br - y2_tl)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def is_blurry(image, threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# ============================================================================
# OBJECT TRACKER CLASS (for a single object)
# ============================================================================

class ObjectTracker:
    """Tracks a single object independently"""
    def __init__(self, obj_name, obj_specs, orb, bf_matcher, K, dist):
        self.obj_name = obj_name
        self.obj_width = obj_specs["width_m"]
        self.obj_height = obj_specs["height_m"]
        self.color = obj_specs["color"]
        
        self.orb = orb
        self.bf = bf_matcher
        self.K = K
        self.dist = dist
        
        self.ref_image = None
        self.ref_kp = None
        self.ref_des = None
        self.plane_ref = None
        self.has_reference = False
        self.stable_bbox_buffer = []
        self.prev_rvec = None
        self.prev_tvec = None
    
    def is_bbox_stable(self, bbox):
        self.stable_bbox_buffer.append(bbox)
        if len(self.stable_bbox_buffer) > STABLE_FRAMES_NEEDED:
            self.stable_bbox_buffer.pop(0)
        if len(self.stable_bbox_buffer) < STABLE_FRAMES_NEEDED:
            return False
        first_bbox = self.stable_bbox_buffer[0]
        for bbox in self.stable_bbox_buffer[1:]:
            if bbox_iou(first_bbox, bbox) < 0.85:
                return False
        return True
    
    def create_reference(self, frame, bbox, logger):
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2].copy()
        if is_blurry(crop, threshold=100):
            logger.warn(f"{self.obj_name}: Frame too blurry")
            return False
        kp_crop, des_crop = self.orb.detectAndCompute(crop, None)
        if des_crop is None or len(kp_crop) < MIN_FEATURES:
            logger.warn(f"{self.obj_name}: Too few features: {0 if des_crop is None else len(kp_crop)}")
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CREATING REFERENCE: {self.obj_name.upper()}")
        logger.info(f"Detected {len(kp_crop)} features")
        
        crop_h, crop_w = crop.shape[:2]
        sx = self.obj_width / crop_w
        sy = self.obj_height / crop_h
        
        plane_ref = []
        valid_kp = []
        valid_des = []
        
        for i, kp in enumerate(kp_crop):
            u, v = kp.pt
            X = u * sx
            Y = -v * sy
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
        
        logger.info(f"✓ Reference created: {len(self.ref_kp)} features")
        logger.info(f"{'='*60}\n")
        return True
    
    def track_pose(self, frame):
        if not self.has_reference:
            return None, "No reference"
        
        kp_live, des_live = self.orb.detectAndCompute(frame, None)
        if des_live is None or len(kp_live) < MIN_FEATURES:
            return None, "No features"
        
        matches = self.bf.knnMatch(self.ref_des, des_live, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < MIN_MATCH_COUNT:
            return None, f"Few matches ({len(good_matches)})"
        
        obj_pts = []
        img_pts = []
        for m in good_matches:
            P3 = self.plane_ref[m.queryIdx]
            if np.any(np.isnan(P3)):
                continue
            pt2d = kp_live[m.trainIdx].pt
            obj_pts.append(P3)
            img_pts.append(pt2d)
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        
        if len(obj_pts) < MIN_PNP_POINTS:
            return None, f"Few PnP points ({len(obj_pts)})"
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, self.dist,
            iterationsCount=100, reprojectionError=8.0,
            confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            return None, "PnP failed"
        
        inliers = inliers.reshape(-1)
        obj_in = obj_pts[inliers]
        img_in = img_pts[inliers]
        
        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        errors = np.linalg.norm(proj - img_in, axis=1)
        mean_err = float(np.mean(errors))
        
        if mean_err < REPROJ_ERROR_THRESH * 2:
            keep_idx = np.where(errors < REPROJ_ERROR_THRESH)[0]
            if len(keep_idx) >= MIN_PNP_POINTS:
                obj_ref = obj_in[keep_idx]
                img_ref = img_in[keep_idx]
                try:
                    success_ref, rvec_ref, tvec_ref = cv2.solvePnP(
                        obj_ref, img_ref, self.K, self.dist,
                        rvec, tvec, True, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success_ref:
                        rvec, tvec = rvec_ref, tvec_ref
                        obj_in, img_in = obj_ref, img_ref
                        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, self.K, self.dist)
                        proj = proj.reshape(-1, 2)
                        errors = np.linalg.norm(proj - img_in, axis=1)
                        mean_err = float(np.mean(errors))
                except cv2.error:
                    pass
        
        self.prev_rvec = rvec.copy()
        self.prev_tvec = tvec.copy()
        
        pose_data = {
            'rvec': rvec, 'tvec': tvec, 'inliers': img_in,
            'n_inliers': len(inliers), 'n_matches': len(good_matches),
            'reproj_error': mean_err
        }
        status = f"Err={mean_err:.2f}px, {len(inliers)} inliers"
        return pose_data, status
    
    def reset(self):
        self.has_reference = False
        self.ref_image = None
        self.ref_kp = None
        self.ref_des = None
        self.plane_ref = None
        self.stable_bbox_buffer.clear()
        self.prev_rvec = None
        self.prev_tvec = None

# ============================================================================
# MULTI-OBJECT TRACKER WITH ROS2
# ============================================================================

class MultiObjectTracker(Node):
    def __init__(self):
        super().__init__('multi_object_tracker_node')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pipeline = None
        self.sink = None
        
        self.get_logger().info("Loading YOLO model...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.get_logger().info("Model loaded!")
        
        self.K, self.dist = load_calibration()
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Create tracker for each object
        self.trackers = {}
        for obj_name, obj_specs in OBJECT_SPECS.items():
            self.trackers[obj_name] = ObjectTracker(
                obj_name, obj_specs, self.orb, self.bf, self.K, self.dist
            )
            self.get_logger().info(f"✓ Tracker created for {obj_name}")
        
        self.frame_count = 0
    
    def publish_tf(self, obj_name, rvec, tvec):
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
    
    def start_camera(self):
        self.get_logger().info("Starting RPi CSI Camera...")
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
            self.get_logger().info("✓ Camera started: 640x480@30fps")
            return True
        except Exception as e:
            self.get_logger().error(f"Camera error: {e}")
            return False
    
    def pull_frame(self, timeout_ns=10_000_000):
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
    
    def draw_axes(self, display, obj_name, rvec, tvec):
        tracker = self.trackers[obj_name]
        axis_len = min(tracker.obj_width, tracker.obj_height) * 0.7
        axis_3d = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, self.K, self.dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)
        
        origin = tuple(imgpts[0])
        x_end = tuple(imgpts[1])
        y_end = tuple(imgpts[2])
        z_end = tuple(imgpts[3])
        
        cv2.line(display, origin, x_end, (0, 0, 255), 3)
        cv2.line(display, origin, y_end, (0, 255, 0), 3)
        cv2.line(display, origin, z_end, (255, 0, 0), 3)
    
    def run(self):
        if not self.start_camera():
            return
        
        self.get_logger().info("\n" + "="*60)
        self.get_logger().info("Multi-Object 6DOF Tracker + ROS2")
        self.get_logger().info("="*60)
        self.get_logger().info(f"Tracking: {', '.join(OBJECT_SPECS.keys())}")
        self.get_logger().info(f"TF Parent Frame: {CAMERA_FRAME}")
        self.get_logger().info("Press 'q' to quit, 'r' to reset all\n")
        
        while rclpy.ok():
            frame = self.pull_frame()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            self.frame_count += 1
            
            if UNDISTORT and self.dist is not None:
                frame = cv2.undistort(frame, self.K, self.dist)
            
            display = frame.copy()
            
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            
            # Collect all detections
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
            
            # Process each object
            tracking_count = 0
            y_text_offset = 30
            
            for obj_name, detection in detections.items():
                tracker = self.trackers[obj_name]
                
                if detection is None:
                    # No detection for this object
                    tracker.stable_bbox_buffer.clear()
                    continue
                
                bbox, conf = detection
                x1, y1, x2, y2 = bbox
                
                # Draw bbox
                if DRAW_BBOX:
                    color = tracker.color if tracker.has_reference else (128, 128, 128)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, f"{obj_name} ({conf:.2f})", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if not tracker.has_reference:
                    # Try to create reference
                    if tracker.is_bbox_stable(bbox):
                        success = tracker.create_reference(frame, bbox, self.get_logger())
                        if not success:
                            tracker.stable_bbox_buffer.clear()
                    else:
                        stable_count = len(tracker.stable_bbox_buffer)
                        cv2.putText(display, f"{obj_name}: Stabilizing {stable_count}/{STABLE_FRAMES_NEEDED}",
                                   (10, y_text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_text_offset += 20
                else:
                    # Track pose
                    pose_data, status = tracker.track_pose(frame)
                    
                    if pose_data is None:
                        cv2.putText(display, f"{obj_name}: Lost - {status}",
                                   (10, y_text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        y_text_offset += 20
                    else:
                        # Publish TF
                        self.publish_tf(obj_name, pose_data['rvec'], pose_data['tvec'])
                        
                        # Draw axes
                        self.draw_axes(display, obj_name, pose_data['rvec'], pose_data['tvec'])
                        
                        # Draw inliers
                        if DRAW_INLIERS:
                            for pt in pose_data['inliers']:
                                cv2.circle(display, (int(pt[0]), int(pt[1])), 3, tracker.color, -1)
                        
                        # Display pose info
                        t = pose_data['tvec'].ravel()
                        cv2.putText(display, f"{obj_name}: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m",
                                   (10, y_text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker.color, 2)
                        y_text_offset += 20
                        
                        tracking_count += 1
            
            # Status overlay
            status_text = f"Tracking: {tracking_count}/{len(OBJECT_SPECS)} | Publishing TF"
            cv2.putText(display, status_text, (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if tracking_count > 0 else (0, 0, 255), 2)
            
            cv2.imshow("Multi-Object 6DOF Tracker", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.get_logger().info("Resetting all references...")
                for tracker in self.trackers.values():
                    tracker.reset()
                self.get_logger().info("✓ All references reset")
            
            rclpy.spin_once(self, timeout_sec=0)
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        self.get_logger().info("Tracking stopped.")

def main():
    rclpy.init()
    tracker = MultiObjectTracker()
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("\nShutdown")
    finally:
        tracker.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()