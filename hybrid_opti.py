#!/usr/bin/env python3
"""
Optimized 5-Object 6DOF Pose Tracker for RPi
- Feature extraction limited to ROIs (Region of Interest)
- YOLO inference skipped for N frames (Inter-frame tracking)
- Optimized ORB parameters
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"
# Force OpenCV to use single thread to prevent fighting with ROS2/GStreamer
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

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
# PERFORMANCE CONFIGURATION
# ============================================================================
YOLO_MODEL_PATH = 'runs/detect/yolov8n_detect_V2/weights/best.pt'
# RECOMMENDATION: Export your model to ONNX for 2-3x speedup on CPU
# yolo export model=best.pt format=onnx opset=12

YOLO_SKIP_FRAMES = 10     # Only run YOLO every N frames
ORB_FEATURES = 500        # Reduced from 2000 for RPi
SEARCH_MARGIN = 30        # Pixel expansion around last known bbox for tracking

# ============================================================================
# STANDARD CONFIGURATION
# ============================================================================
ARUCO_DICT = cv2.aruco.DICT_4X4_50

ARUCO_IDS = { "phone": 0, "estop": 1 }
ARUCO_MARKER_SIZES = { "phone": 0.0365, "estop": 0.0365 }

OBJECT_SPECS = {
    "card_game": { "width_m": 0.093, "height_m": 0.115, "color": (0, 255, 0), "method": "feature" },
    "circuit_board": { "width_m": 0.210, "height_m": 0.210, "color": (255, 0, 0), "method": "feature" },
    "notebook": { "width_m": 0.147, "height_m": 0.209, "color": (0, 0, 255), "method": "feature" },
    "phone": { "color": (255, 255, 0), "method": "aruco", "aruco_id": 0, "marker_size_m": 0.0365 },
    "estop": { "color": (255, 0, 255), "method": "aruco", "aruco_id": 1, "marker_size_m": 0.0365 },
}

MIN_FEATURES = 10
MIN_MATCH_COUNT = 8
MIN_PNP_POINTS = 6
MIN_PNP_INLIERS = 6
REPROJ_ERROR_THRESH = 4.0
CAMERA_FRAME = "camera_link_G4"
DRAW_VISUALS = True # Set False for max FPS

# ============================================================================
# FEATURE TRACKER CLASS (Optimized)
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
        
        # Reference data
        self.ref_kp = None
        self.ref_des = None
        self.plane_ref = None
        self.has_reference = False
        
        # Tracking state
        self.last_bbox = None  # [x1, y1, x2, y2]
        self.prev_rvec = None
        self.prev_tvec = None
        self.missed_frames = 0
    
    def create_reference(self, frame, bbox, logger):
        x1, y1, x2, y2 = bbox
        # Add slight padding for reference creation
        h, w = frame.shape[:2]
        x1 = max(0, x1 - 10); y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10); y2 = min(h, y2 + 10)
        
        crop = frame[y1:y2, x1:x2].copy()
        kp_crop, des_crop = self.orb.detectAndCompute(crop, None)
        
        if des_crop is None or len(kp_crop) < MIN_FEATURES:
            return False
            
        # Scale physical dimensions
        crop_h, crop_w = crop.shape[:2]
        sx = self.obj_width / crop_w
        sy = self.obj_height / crop_h
        
        plane_ref = []
        valid_kp = []
        valid_des = []
        
        # We store keypoints relative to the Top-Left of the object
        # This helps in PnP later
        for i, kp in enumerate(kp_crop):
            u, v = kp.pt
            X = u * sx
            Y = -v * sy
            Z = 0.0
            plane_ref.append([X, Y, Z])
            valid_kp.append(kp)
            valid_des.append(des_crop[i])
        
        self.ref_kp = valid_kp
        self.ref_des = np.array(valid_des)
        self.plane_ref = np.array(plane_ref, dtype=np.float32)
        self.has_reference = True
        self.last_bbox = bbox # Initialize tracking window
        logger.info(f"Ref created for {self.obj_name} with {len(valid_kp)} features")
        return True

    def track_pose(self, frame, yolo_bbox=None):
        """
        Optimized tracking:
        1. If yolo_bbox is provided, use it.
        2. If not, use self.last_bbox expanded by margin.
        3. Crop image -> Detect ORB -> Shift Keypoints back to global space -> PnP.
        """
        if not self.has_reference:
            return None, "No Ref"

        # Determine Region of Interest (ROI)
        roi_box = None
        h, w = frame.shape[:2]

        if yolo_bbox is not None:
            roi_box = yolo_bbox
            self.missed_frames = 0
        elif self.last_bbox is not None:
            # Expand last known position
            lx1, ly1, lx2, ly2 = self.last_bbox
            roi_box = [
                max(0, lx1 - SEARCH_MARGIN),
                max(0, ly1 - SEARCH_MARGIN),
                min(w, lx2 + SEARCH_MARGIN),
                min(h, ly2 + SEARCH_MARGIN)
            ]
        else:
            # Lost track completely, search full frame (expensive)
            roi_box = [0, 0, w, h]

        # CROP AND DETECT
        rx1, ry1, rx2, ry2 = roi_box
        if rx2 <= rx1 or ry2 <= ry1: return None, "Invalid ROI"
        
        crop = frame[ry1:ry2, rx1:rx2]
        kp_live, des_live = self.orb.detectAndCompute(crop, None)

        if des_live is None or len(kp_live) < MIN_FEATURES:
            self.missed_frames += 1
            return None, "No Features"

        # Match
        matches = self.bf.knnMatch(self.ref_des, des_live, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < MIN_MATCH_COUNT:
            self.missed_frames += 1
            return None, "Low Matches"

        # Prepare for PnP
        obj_pts = []
        img_pts = []

        for m in good_matches:
            obj_pts.append(self.plane_ref[m.queryIdx])
            # CRITICAL: Shift crop coordinates back to global frame
            local_pt = kp_live[m.trainIdx].pt
            global_pt = (local_pt[0] + rx1, local_pt[1] + ry1)
            img_pts.append(global_pt)

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        # Solve PnP
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_pts, img_pts, self.K, self.dist,
                iterationsCount=50, # Reduced from 100
                reprojectionError=REPROJ_ERROR_THRESH,
                confidence=0.95, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception:
            return None, "PnP Error"

        if not success or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            self.missed_frames += 1
            return None, "PnP Failed"

        # Update tracking state
        self.prev_rvec = rvec
        self.prev_tvec = tvec
        
        # Update Tracking ROI based on inliers (keep the bounding box tight around the object)
        # This allows us to track without YOLO for many frames
        inlier_pts = img_pts[inliers.ravel()]
        if len(inlier_pts) > 0:
            min_x = int(np.min(inlier_pts[:, 0]))
            max_x = int(np.max(inlier_pts[:, 0]))
            min_y = int(np.min(inlier_pts[:, 1]))
            max_y = int(np.max(inlier_pts[:, 1]))
            self.last_bbox = [min_x, min_y, max_x, max_y]
            self.missed_frames = 0

        pose_data = {'rvec': rvec, 'tvec': tvec}
        return pose_data, "Tracking"

    def reset(self):
        self.has_reference = False
        self.last_bbox = None

# ============================================================================
# MAIN NODE
# ============================================================================

class FastTracker(Node):
    def __init__(self):
        super().__init__('fast_tracker_node')
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Camera Setup
        self.K = np.array(CAMERA_MATRIX, dtype=np.float32)
        self.dist = np.array(DIST_COEFFS, dtype=np.float32) if DIST_COEFFS is not None else None
        
        self.pipeline = None
        self.sink = None
        self.start_camera()

        # Initialize Models
        self.get_logger().info(f"Loading YOLO... (Skipping {YOLO_SKIP_FRAMES} frames)")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Initialize Feature Detectors
        # FastThreshold=20 makes ORB faster by ignoring low contrast points
        self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=20) 
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Initialize ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Create Trackers
        self.trackers = {}
        for name, specs in OBJECT_SPECS.items():
            if specs["method"] == "feature":
                self.trackers[name] = FeatureTracker(name, specs, self.orb, self.bf, self.K, self.dist)

        self.frame_count = 0

    def start_camera(self):
        Gst.init(None)
        # Reduced buffer to 1 to ensure lowest latency
        gst_str = (
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        self.pipeline = Gst.parse_launch(gst_str)
        self.sink = self.pipeline.get_by_name("sink")
        self.pipeline.set_state(Gst.State.PLAYING)

    def pull_frame(self):
        sample = self.sink.emit("try-pull-sample", 5000000) # 5ms timeout
        if not sample: return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        arr = np.ndarray(
            (caps.get_value("height"), caps.get_value("width"), 3),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8
        )
        return arr

    def publish_tf(self, obj_name, rvec, tvec):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = CAMERA_FRAME
        t.child_frame_id = f"{obj_name}_frame"
        t.transform.translation.x = float(tvec[0])
        t.transform.translation.y = float(tvec[1])
        t.transform.translation.z = float(tvec[2])
        
        rot_mat, _ = cv2.Rodrigues(rvec)
        q = tf_transformations.quaternion_from_matrix(
            np.vstack([np.hstack([rot_mat, [[0],[0],[0]]]), [0,0,0,1]])
        )
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def run(self):
        self.get_logger().info("Starting Fast Tracking Loop")
        
        while rclpy.ok():
            frame = self.pull_frame()
            if frame is None:
                continue

            self.frame_count += 1
            display = frame.copy() if DRAW_VISUALS else None
            
            # ==========================
            # 1. YOLO DETECTION (INTERMITTENT)
            # ==========================
            run_yolo = (self.frame_count % YOLO_SKIP_FRAMES == 0) or (self.frame_count < 10)
            yolo_results = {}

            if run_yolo:
                results = self.yolo_model(frame, verbose=False, stream=True)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        cls_name = self.yolo_model.names[int(box.cls[0])]
                        if cls_name in OBJECT_SPECS and float(box.conf[0]) > 0.6:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            yolo_results[cls_name] = xyxy
                            if DRAW_VISUALS:
                                cv2.rectangle(display, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255,255,255), 1)

            # ==========================
            # 2. FEATURE TRACKERS
            # ==========================
            for name, tracker in self.trackers.items():
                # Logic: Use YOLO bbox if available, else let tracker use internal memory
                bbox = yolo_results.get(name)
                
                # If we have a new YOLO box, see if we need to initialize reference
                if bbox is not None and not tracker.has_reference:
                    tracker.create_reference(frame, bbox, self.get_logger())
                    continue # Skip tracking this frame, just setup

                # Run Tracking (PnP)
                # If bbox is None, track_pose uses its own 'last_bbox'
                pose, status = tracker.track_pose(frame, yolo_bbox=bbox)

                if pose:
                    self.publish_tf(name, pose['rvec'], pose['tvec'])
                    if DRAW_VISUALS:
                        # Simple axis drawing for speed
                        cv2.drawFrameAxes(display, self.K, self.dist, pose['rvec'], pose['tvec'], 0.1)
                        cv2.putText(display, name, (int(tracker.last_bbox[0]), int(tracker.last_bbox[1])), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker.color, 2)
                elif DRAW_VISUALS and tracker.has_reference:
                     cv2.putText(display, f"{name}: {status}", (10, 30 + list(self.trackers).index(name)*20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # ==========================
            # 3. ARUCO TRACKERS (Always Run - Fast)
            # ==========================
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            
            if ids is not None:
                flat_ids = ids.flatten()
                for name, specs in OBJECT_SPECS.items():
                    if specs["method"] == "aruco" and specs["aruco_id"] in flat_ids:
                        idx = np.where(flat_ids == specs["aruco_id"])[0][0]
                        c = corners[idx]
                        
                        # Use IPPE_SQUARE for faster PnP on planar markers
                        ret, rvec, tvec = cv2.solvePnP(
                            np.array([[-0.5,0.5,0], [0.5,0.5,0], [0.5,-0.5,0], [-0.5,-0.5,0]]) * specs["marker_size_m"],
                            c, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                        if ret:
                            self.publish_tf(name, rvec, tvec)
                            if DRAW_VISUALS:
                                cv2.drawFrameAxes(display, self.K, self.dist, rvec, tvec, 0.05)

            # ==========================
            # 4. UI & ROS
            # ==========================
            if DRAW_VISUALS:
                cv2.imshow("FastTracker", display)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            rclpy.spin_once(self, timeout_sec=0)

        self.pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

def main():
    rclpy.init()
    node = FastTracker()
    try: node.run()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()