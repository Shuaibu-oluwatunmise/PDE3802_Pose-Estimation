"""
Homography Suitability Tester
Tests how well different objects work for homography-based tracking
Shows real-time metrics to help you choose the best objects
"""

import cv2
import numpy as np
import time

class HomographySuitabilityTester:
    def __init__(self, camera_index=1):
        self.cap = None
        self.camera_index = camera_index
        
        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Reference state
        self.ref_image = None
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_name = "Object"
        self.calibrated = False
        
        # Tracking history for statistics
        self.history = {
            "good_matches": [],
            "inlier_ratios": [],
            "success_count": 0,
            "fail_count": 0,
        }
        
        print("✓ Homography Suitability Tester initialized")
    
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
    
    def calibrate_reference(self, frame, roi_coords=None):
        """Calibrate reference image"""
        if roi_coords is not None:
            x1, y1, x2, y2 = roi_coords
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 50:
            print(f"✗ Not enough features: {len(keypoints) if keypoints else 0}")
            return False
        
        self.ref_image = gray
        self.ref_keypoints = keypoints
        self.ref_descriptors = descriptors
        self.calibrated = True
        
        # Reset history
        self.history = {
            "good_matches": [],
            "inlier_ratios": [],
            "success_count": 0,
            "fail_count": 0,
        }
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION COMPLETE")
        print(f"{'='*60}")
        print(f"Reference features: {len(keypoints)}")
        print(f"{'='*60}\n")
        
        return True
    
    def test_homography(self, frame):
        """Test homography matching and return metrics"""
        if not self.calibrated:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 10:
            return {
                "status": "fail",
                "reason": "not_enough_keypoints",
                "current_kp": len(keypoints) if keypoints else 0,
            }
        
        # Match features
        matches = self.bf_matcher.knnMatch(self.ref_descriptors, descriptors, k=2)
        
        # Ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            return {
                "status": "fail",
                "reason": "not_enough_matches",
                "current_kp": len(keypoints),
                "good_matches": len(good_matches),
            }
        
        # Try homography
        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
        
        if H is None or mask is None:
            return {
                "status": "fail",
                "reason": "ransac_failed",
                "current_kp": len(keypoints),
                "good_matches": len(good_matches),
            }
        
        inliers = mask.ravel().tolist()
        inlier_count = sum(inliers)
        inlier_ratio = inlier_count / len(inliers)
        
        # Get corner transformation
        h_ref, w_ref = self.ref_image.shape
        ref_corners = np.array([[0, 0], [w_ref, 0], [w_ref, h_ref], [0, h_ref]], dtype=np.float32).reshape(-1, 1, 2)
        corners_frame = cv2.perspectiveTransform(ref_corners, H).reshape(-1, 2)
        
        # Update history
        self.history["good_matches"].append(len(good_matches))
        self.history["inlier_ratios"].append(inlier_ratio)
        if inlier_ratio >= 0.25:
            self.history["success_count"] += 1
        else:
            self.history["fail_count"] += 1
        
        # Keep only last 100 samples
        if len(self.history["good_matches"]) > 100:
            self.history["good_matches"] = self.history["good_matches"][-100:]
            self.history["inlier_ratios"] = self.history["inlier_ratios"][-100:]
        
        return {
            "status": "success",
            "current_kp": len(keypoints),
            "ref_kp": len(self.ref_keypoints),
            "good_matches": len(good_matches),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "corners": corners_frame,
            "H": H,
        }
    
    def get_statistics(self):
        """Get statistics from tracking history"""
        if len(self.history["good_matches"]) == 0:
            return None
        
        return {
            "avg_good_matches": np.mean(self.history["good_matches"]),
            "min_good_matches": np.min(self.history["good_matches"]),
            "max_good_matches": np.max(self.history["good_matches"]),
            "avg_inlier_ratio": np.mean(self.history["inlier_ratios"]),
            "min_inlier_ratio": np.min(self.history["inlier_ratios"]),
            "max_inlier_ratio": np.max(self.history["inlier_ratios"]),
            "success_rate": self.history["success_count"] / (self.history["success_count"] + self.history["fail_count"]) * 100,
            "total_samples": len(self.history["good_matches"]),
        }
    
    def draw_3d_axes(self, frame, H, length=80):
        """Draw 3D axes at the center of the tracked object"""
        h_ref, w_ref = self.ref_image.shape
        
        # Center of reference image
        center_x = w_ref / 2.0
        center_y = h_ref / 2.0
        
        # Define 3D axes points in reference frame (Z points out of plane)
        axis_points_ref = np.float32([
            [center_x, center_y],              # Origin
            [center_x + length, center_y],     # X-axis (Red)
            [center_x, center_y + length],     # Y-axis (Green)
        ]).reshape(-1, 1, 2)
        
        # Add Z-axis by using homography twice with offset
        # We'll draw Z by computing a perpendicular direction
        
        # Transform axes to current frame
        axis_points_frame = cv2.perspectiveTransform(axis_points_ref, H).reshape(-1, 2)
        
        origin = tuple(axis_points_frame[0].astype(int))
        x_end = tuple(axis_points_frame[1].astype(int))
        y_end = tuple(axis_points_frame[2].astype(int))
        
        # Draw X-axis (Red)
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 3, tipLength=0.3)
        cv2.putText(frame, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw Y-axis (Green)
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(frame, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw Z-axis (Blue) - perpendicular to XY plane
        # Estimate Z direction based on cross product visualization
        z_end = (origin[0], origin[1] - int(length * 0.8))  # Point upward
        cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), 3, tipLength=0.3)
        cv2.putText(frame, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw origin circle
        cv2.circle(frame, origin, 8, (255, 255, 255), -1)
        
        return frame
    
    def draw_results(self, frame, result):
        """Draw tracking results on frame"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        if not self.calibrated:
            # Not calibrated - show instructions
            cv2.putText(display, "NOT CALIBRATED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, "Press SPACE to calibrate from current frame", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "Press 'c' to calibrate from center ROI", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center ROI guide
            roi_w, roi_h = 300, 300
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            x2 = x1 + roi_w
            y2 = y1 + roi_h
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(display, "Place object here", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            return display
        
        # Calibrated - show tracking results
        if result is None:
            cv2.putText(display, "Processing...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return display
        
        # Draw based on status
        if result["status"] == "fail":
            color = (0, 0, 255)  # Red
            status_text = f"FAIL: {result['reason']}"
            
            cv2.putText(display, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if "current_kp" in result:
                cv2.putText(display, f"Current KP: {result['current_kp']}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if "good_matches" in result:
                cv2.putText(display, f"Good matches: {result['good_matches']}", (10, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        else:  # success
            # Draw tracked corners
            corners = result["corners"].astype(int)
            cv2.polylines(display, [corners], True, (0, 255, 0), 3)
            
            # Draw 3D axes at center
            display = self.draw_3d_axes(display, result["H"])
            
            # Color based on inlier ratio quality
            inlier_ratio = result["inlier_ratio"]
            if inlier_ratio >= 0.7:
                color = (0, 255, 0)      # Green - Excellent
                quality = "EXCELLENT"
            elif inlier_ratio >= 0.5:
                color = (0, 255, 255)    # Yellow - Good
                quality = "GOOD"
            elif inlier_ratio >= 0.3:
                color = (0, 165, 255)    # Orange - Fair
                quality = "FAIR"
            else:
                color = (0, 0, 255)      # Red - Poor
                quality = "POOR"
            
            # Status
            cv2.putText(display, f"TRACKING: {quality}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Metrics
            y_offset = 60
            cv2.putText(display, f"Ref KP: {result['ref_kp']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display, f"Cur KP: {result['current_kp']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display, f"Good matches: {result['good_matches']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display, f"Inliers: {result['inlier_count']}/{result['good_matches']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(display, f"Inlier ratio: {inlier_ratio:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show statistics
        stats = self.get_statistics()
        if stats:
            y_offset = h - 150
            cv2.rectangle(display, (5, y_offset - 5), (350, h - 5), (0, 0, 0), -1)
            cv2.rectangle(display, (5, y_offset - 5), (350, h - 5), (255, 255, 255), 2)
            
            cv2.putText(display, "=== STATISTICS ===", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(display, f"Samples: {stats['total_samples']}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
            
            cv2.putText(display, f"Avg matches: {stats['avg_good_matches']:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
            
            cv2.putText(display, f"Avg inlier ratio: {stats['avg_inlier_ratio']:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
            
            cv2.putText(display, f"Min inlier ratio: {stats['min_inlier_ratio']:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
            
            # Success rate with color
            success_rate = stats['success_rate']
            if success_rate >= 80:
                rate_color = (0, 255, 0)
            elif success_rate >= 60:
                rate_color = (0, 255, 255)
            else:
                rate_color = (0, 0, 255)
            
            cv2.putText(display, f"Success rate: {success_rate:.1f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, rate_color, 1)
            y_offset += 25
            
            # Suitability verdict
            if stats['avg_inlier_ratio'] >= 0.7 and success_rate >= 80:
                verdict = "EXCELLENT for homography!"
                verdict_color = (0, 255, 0)
            elif stats['avg_inlier_ratio'] >= 0.5 and success_rate >= 70:
                verdict = "GOOD for homography"
                verdict_color = (0, 255, 255)
            elif stats['avg_inlier_ratio'] >= 0.3 and success_rate >= 50:
                verdict = "FAIR - might be unstable"
                verdict_color = (0, 165, 255)
            else:
                verdict = "POOR - try another object"
                verdict_color = (0, 0, 255)
            
            cv2.putText(display, verdict, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, verdict_color, 2)
        
        return display
    
    def run(self):
        """Main testing loop"""
        print("\n" + "="*60)
        print("HOMOGRAPHY SUITABILITY TESTER")
        print("="*60)
        print("\nThis tool helps you evaluate objects for homography tracking")
        print("\nControls:")
        print("  SPACE - Calibrate from full frame")
        print("  'c'   - Calibrate from center ROI (recommended)")
        print("  'r'   - Reset calibration")
        print("  's'   - Save current frame")
        print("  ESC   - Exit")
        print("\nHow to use:")
        print("1. Place object in center ROI")
        print("2. Press 'c' to calibrate")
        print("3. Move object around, rotate it, change distance")
        print("4. Watch the statistics - aim for:")
        print("   - Avg inlier ratio > 0.5 (Good)")
        print("   - Success rate > 70%")
        print("5. Press 'r' to test another object")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return
        
        saved_count = 0
        
        try:
            while True:
                frame = self.pull_frame()
                if frame is None:
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue
                
                # Test homography if calibrated
                result = None
                if self.calibrated:
                    result = self.test_homography(frame)
                
                # Draw results
                display = self.draw_results(frame, result)
                
                cv2.imshow('Homography Suitability Tester', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                
                elif key == ord(' '):  # Space - calibrate full frame
                    print("Calibrating from full frame...")
                    self.calibrate_reference(frame)
                
                elif key == ord('c'):  # 'c' - calibrate center ROI
                    print("Calibrating from center ROI...")
                    h, w = frame.shape[:2]
                    roi_w, roi_h = 300, 300
                    x1 = (w - roi_w) // 2
                    y1 = (h - roi_h) // 2
                    x2 = x1 + roi_w
                    y2 = y1 + roi_h
                    self.calibrate_reference(frame, roi_coords=[x1, y1, x2, y2])
                
                elif key == ord('r'):  # 'r' - reset
                    self.calibrated = False
                    self.ref_image = None
                    self.ref_keypoints = None
                    self.ref_descriptors = None
                    print("✓ Calibration reset - ready for new object")
                
                elif key == ord('s'):  # 's' - save
                    filename = f"homography_test_{saved_count:03d}.jpg"
                    cv2.imwrite(filename, display)
                    saved_count += 1
                    print(f"✓ Saved: {filename}")
                    
                    # Also print statistics
                    stats = self.get_statistics()
                    if stats:
                        print(f"   Stats at save:")
                        print(f"   - Avg inlier ratio: {stats['avg_inlier_ratio']:.2f}")
                        print(f"   - Min inlier ratio: {stats['min_inlier_ratio']:.2f}")
                        print(f"   - Success rate: {stats['success_rate']:.1f}%")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    camera_index = 1
    tester = HomographySuitabilityTester(camera_index)
    tester.run()


if __name__ == "__main__":
    main()