"""
Planar Tracking Module (Modular Component)
Handles ORB-based homography tracking and "Best-Shot" autocalibration
for planar objects (Notebook, Game Box, Screw Mat).
"""

import cv2
import numpy as np

class PlanarTracker:
    def __init__(self, obj_configs, bf_matcher, orb):
        self.bf_matcher = bf_matcher
        self.orb = orb
        self.targets = {}

        # 🟢 Initialize target state and 3D points based on measurements
        for obj_name, cfg in obj_configs.items():                
            width_mm = cfg["width_mm"]
            height_mm = cfg["height_mm"]                
            half_w = width_mm / 2.0                
            half_h = height_mm / 2.0                

            # Planar points, Z=0. Center of surface is [0,0,0]
            # Coordinates: Top-Left, Top-Right, Bottom-Right, Bottom-Left
            object_points_3d = np.array([                
                [-half_w,  half_h, 0],   
                [ half_w,  half_h, 0],   
                [ half_w, -half_h, 0],   
                [-half_w, -half_h, 0],   
            ], dtype=np.float32)                

            self.targets[obj_name] = {                
                "object_points_3d": object_points_3d,                
                "min_matches": cfg.get("min_matches", 10),                
                "min_inlier_ratio": cfg.get("min_inlier_ratio", 0.5),                
                
                # Reference state
                "ref_image": None,                
                "ref_keypoints": None,                
                "ref_descriptors": None,                
                "calibrated": False,                
                
                # Autocalibration buffer
                "calib_buffer": [],                
                "calib_buffer_size": 3, 
            }                

    def get_target_state(self, obj_name):
        """Returns the tracking/calibration state for a specific target."""
        return self.targets.get(obj_name)

    # ------------------------------------------------------------------
    # 🟢 Autocalibration: "Best Shot" buffering
    # ------------------------------------------------------------------
    def accumulate_and_maybe_calibrate(self, frame, obj_name, bbox):
        """
        Collects candidate ROIs over multiple frames and calibrates once
        we have enough, using the ROI with the most keypoints.
        Returns True iff calibration was performed successfully.
        """
        state = self.targets[obj_name]
        x1, y1, x2, y2 = bbox                
        
        # Crop the candidate ROI immediately to save memory and process
        roi = frame[y1:y2, x1:x2].copy()                
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)                
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)                

        if descriptors is None or len(keypoints) < 10:                
            # Not a useful candidate (e.g., blur/no texture); skip
            return False                

        # Add candidate to buffer
        state["calib_buffer"].append({                
            "roi": roi,                
            "n_kp": len(keypoints),                
        })                

        # Not enough candidates to pick the best yet
        if len(state["calib_buffer"]) < state["calib_buffer_size"]:                
            print(f"Buffered candidate for {obj_name} ({len(state['calib_buffer'])}/{state['calib_buffer_size']})")                
            return False                

        # --- BUFFER FULL: Perform Best-Frame Selection ---
        print(f"Buffer full for {obj_name}. Selecting best frame...")                
        
        # Pick the candidate with the absolute maximum number of keypoints
        best_candidate = max(state["calib_buffer"], key=lambda c: c["n_kp"])                
        roi_best = best_candidate["roi"]                

        # Run final calibration on this best ROI
        success = self._finalize_calibration(roi_best, obj_name)                

        # Clear buffer for next time (or retry if failed)
        state["calib_buffer"].clear()                

        return success                

    def _finalize_calibration(self, roi, obj_name):                
        """Performs actual feature extraction on the best candidate ROI."""
        state = self.targets[obj_name]
        print(f"✓ CALIBRATING REFERENCE: {obj_name.upper()}")

        state["ref_image"] = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)                
        state["ref_keypoints"], state["ref_descriptors"] = self.orb.detectAndCompute(                
            state["ref_image"], None                
        )                

        if state["ref_descriptors"] is None or len(state["ref_keypoints"]) < 10:                
            print("ERROR: Not enough features in final candidate! Retrying window.")                
            state["calibrated"] = False                
            return False                

        # Reset state for temporal smoothing
        state["rvec_smooth"] = None                
        state["tvec_smooth"] = None                

        print(f"✓ Detected {len(state['ref_keypoints'])} features")                
        state["calibrated"] = True                
        return True                

    # ------------------------------------------------------------------
    # 🟢 Tracking: ORB Matcher + Homography
    # ------------------------------------------------------------------
    def track_object_homography(self, frame, obj_name, bbox):                
        """
        Takes reference features and matches them to the current frame box.
        Returns 2D corners in the full frame coordinates, or None.
        """
        state = self.targets[obj_name]                

        # Sanity check
        if not state["calibrated"] or state["ref_descriptors"] is None:                
            return None                

        min_matches = state["min_matches"]                
        min_inlier_ratio = state["min_inlier_ratio"]                

        x1_raw, y1_raw, x2_raw, y2_raw = bbox                

        # Pad the box a bit to include more context and reduce jitter                
        pad = 10                
        h, w, _ = frame.shape                
        x1 = max(0, x1_raw - pad)                
        y1 = max(0, y1_raw - pad)                
        x2 = min(w - 1, x2_raw + pad)                
        y2 = min(h - 1, y2_raw + pad)                

        # Extract ROI features                
        roi = frame[y1:y2, x1:x2]                
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)                
        keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)                

        if descriptors is None or len(keypoints) < 10:                
            return None                

        # Perform matching
        matches = self.bf_matcher.knnMatch(state["ref_descriptors"], descriptors, k=2)                

        # Lowe's ratio test (standard 0.75)
        good_matches = []                
        for m_n in matches:                
            if len(m_n) < 2:                
                continue                
            m, n = m_n                
            if m.distance < 0.75 * n.distance:                
                good_matches.append(m)                

        # Minimum matches check
        if len(good_matches) < min_matches:                
            return None                

        src_pts = np.float32(                
            [state["ref_keypoints"][m.queryIdx].pt for m in good_matches]                
        ).reshape(-1, 1, 2)                
        dst_pts = np.float32(                
            [keypoints[m.trainIdx].pt for m in good_matches]                
        ).reshape(-1, 1, 2)                

        # Find homography via RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)                
        if H is None or mask is None:                
            return None                

        # RANSAC inlier ratio check
        inliers = mask.ravel().tolist()                
        inlier_ratio = sum(inliers) / len(inliers)                
        if inlier_ratio < min_inlier_ratio:                
            return None                

        # Transform reference corners to the current frame box ROI
        h_ref, w_ref = state["ref_image"].shape                
        ref_corners = np.array(                
            [[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32                
        ).reshape(-1, 1, 2)                

        corners_roi = cv2.perspectiveTransform(ref_corners, H)                
        corners_roi = corners_roi.reshape(-1, 2)                

        # Transform ROI corners to full frame coordinates
        corners_frame = corners_roi.copy()                
        corners_frame[:, 0] += x1                
        corners_frame[:, 1] += y1                

        return corners_frame