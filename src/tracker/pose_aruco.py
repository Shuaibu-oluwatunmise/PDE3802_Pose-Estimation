"""
ArUco Tracking Module (Modular Component)
Handles detection and 6DOF pose estimation for objects with ArUco tags
(Wallet, Headset).
"""

import cv2
import numpy as np

class ArucoTracker:
    def __init__(self, camera_matrix, dist_coeffs):                
        self.K = camera_matrix
        self.D = dist_coeffs
        
        # 🟢 YOUR CONFIG: Derived from generator images
        # ID 0: Headset, 40mm
        # ID 1: Wallet,  60mm
        self.marker_configs = {                
            0: {"name": "headset", "size_mm": 40.0},                
            1: {"name": "wallet",  "size_mm": 60.0},                
        }                

        # Assessment uses 4x4 dictionary based on generator images
        # Using _50 dictionary is efficient for a small number of markers
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)                
        self.aruco_params = cv2.aruco.DetectorParameters()                

    def estimate_poses(self, frame):                
        """                
        Detects configured ArUco markers and estimates 6DOF poses.                
        Returns dict: name -> {"rvec": rvec, "tvec": tvec}                
        """                
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                
        
        # Detect candidates
        corners, ids, rejected = cv2.aruco.detectMarkers(                
            gray, self.aruco_dict, parameters=self.aruco_params                
        )                

        results = {}                
        
        if ids is not None:                
            # Iterate through detected markers                
            for i, marker_id in enumerate(ids.ravel()):                
                if marker_id in self.marker_configs:                
                    config = self.marker_configs[marker_id]                
                    
                    # 🟢 Standard ArUco Pose Estimation:
                    # Takes 2D corners, 3D marker size, K, and D.
                    # Returns rvecs (rotation) and tvecs (translation).
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(                
                        corners[i], config["size_mm"], self.K, self.D                
                    )                
                    
                    # rvecs/tvecs returned as arrays of arrays, flatten them
                    # to standard (3, 1) vectors.
                    results[config["name"]] = {                
                        "rvec": rvecs[0].reshape(3, 1),                
                        "tvec": tvecs[0].reshape(3, 1)                
                    }                
                    
                    # Visualization (Draws on the frame in-place) 🟢
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)                
                    cv2.aruco.drawAxis(frame, self.K, self.D, 
                                       rvecs[0], tvecs[0], 
                                       config["size_mm"] * 0.5) # Axis length

        return results