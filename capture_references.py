"""
Reference Image Capture Utility
Capture and save high-quality reference images for homography tracking
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

class ReferenceImageCapture:
    def __init__(self, camera_index=1, output_dir="reference_images"):
        self.camera_index = camera_index
        self.output_dir = output_dir
        self.cap = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # ORB for feature detection
        self.orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
        
        print(f"✓ Reference Image Capture initialized")
        print(f"✓ Output directory: {output_dir}")
    
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
    
    def analyze_frame(self, frame):
        """Analyze frame quality for reference capture"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None:
            return {
                "suitable": False,
                "keypoint_count": 0,
                "reason": "No features detected",
                "quality": "POOR"
            }
        
        n_kp = len(keypoints)
        
        # Assess quality
        if n_kp >= 1500:
            quality = "EXCELLENT"
            suitable = True
        elif n_kp >= 1000:
            quality = "GOOD"
            suitable = True
        elif n_kp >= 500:
            quality = "FAIR"
            suitable = True
        else:
            quality = "POOR"
            suitable = False
        
        return {
            "suitable": suitable,
            "keypoint_count": n_kp,
            "quality": quality,
            "keypoints": keypoints,
            "descriptors": descriptors
        }
    
    def save_reference(self, frame, object_name, dimensions_mm):
        """Save reference image with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Analyze frame
        analysis = self.analyze_frame(frame)
        
        if not analysis["suitable"]:
            print(f"✗ Frame quality too low: {analysis['quality']}")
            return False
        
        # Create safe filename
        safe_name = object_name.replace(" ", "_").lower()
        base_filename = f"{safe_name}_{timestamp}"
        
        # Save image
        image_path = os.path.join(self.output_dir, f"{base_filename}.jpg")
        cv2.imwrite(image_path, frame)
        
        # Save metadata
        metadata = {
            "object_name": object_name,
            "dimensions_mm": {
                "width": dimensions_mm[0],
                "height": dimensions_mm[1]
            },
            "timestamp": timestamp,
            "keypoint_count": analysis["keypoint_count"],
            "quality": analysis["quality"],
            "image_shape": frame.shape[:2],  # (height, width)
            "image_file": f"{base_filename}.jpg"
        }
        
        metadata_path = os.path.join(self.output_dir, f"{base_filename}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save keypoints and descriptors separately
        features_path = os.path.join(self.output_dir, f"{base_filename}_features.npz")
        
        # Convert keypoints to saveable format
        kp_data = np.array([(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id) 
                            for kp in analysis["keypoints"]])
        
        np.savez_compressed(features_path, 
                           keypoints=kp_data,
                           descriptors=analysis["descriptors"])
        
        print(f"\n{'='*60}")
        print(f"✓ REFERENCE SAVED: {object_name}")
        print(f"{'='*60}")
        print(f"Image: {image_path}")
        print(f"Metadata: {metadata_path}")
        print(f"Features: {features_path}")
        print(f"Quality: {analysis['quality']}")
        print(f"Keypoints: {analysis['keypoint_count']}")
        print(f"Dimensions: {dimensions_mm[0]}x{dimensions_mm[1]}mm")
        print(f"{'='*60}\n")
        
        return True
    
    def run(self):
        """Main capture loop"""
        print("\n" + "="*60)
        print("REFERENCE IMAGE CAPTURE UTILITY")
        print("="*60)
        print("\nThis tool captures high-quality reference images")
        print("\nWorkflow:")
        print("1. Place object in center of frame")
        print("2. Ensure good lighting and focus")
        print("3. Hold steady")
        print("4. Press SPACE when ready")
        print("5. Enter object name and dimensions")
        print("\nControls:")
        print("  SPACE - Capture reference")
        print("  'f'   - Toggle freeze frame")
        print("  ESC   - Exit")
        print("\nTips:")
        print("  - Fill 30-50% of frame")
        print("  - Use flat, even lighting")
        print("  - Aim for 1000+ keypoints (shown on screen)")
        print("  - Avoid motion blur")
        print("="*60 + "\n")
        
        if not self.start_camera():
            return
        
        frozen = False
        frozen_frame = None
        
        try:
            while True:
                if not frozen:
                    ret, frame = self.cap.read()
                    if not ret:
                        continue
                else:
                    frame = frozen_frame.copy()
                
                display = frame.copy()
                
                # Analyze current frame
                analysis = self.analyze_frame(frame)
                
                # Draw center guide
                h, w = display.shape[:2]
                center_x, center_y = w // 2, h // 2
                box_size = min(w, h) // 2
                x1 = center_x - box_size // 2
                y1 = center_y - box_size // 2
                x2 = center_x + box_size // 2
                y2 = center_y + box_size // 2
                
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(display, "Place object here", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show quality metrics
                quality = analysis["quality"]
                kp_count = analysis["keypoint_count"]
                
                if quality == "EXCELLENT":
                    color = (0, 255, 0)
                elif quality == "GOOD":
                    color = (0, 255, 255)
                elif quality == "FAIR":
                    color = (0, 165, 255)
                else:
                    color = (0, 0, 255)
                
                cv2.putText(display, f"Quality: {quality}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"Keypoints: {kp_count}", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if analysis["suitable"]:
                    cv2.putText(display, "Press SPACE to capture", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, analysis.get("reason", "Improve frame quality"), (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if frozen:
                    cv2.putText(display, "FROZEN - Press 'f' to unfreeze", (10, h-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                cv2.imshow('Reference Image Capture', display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                
                elif key == ord(' '):  # SPACE - capture
                    if not analysis["suitable"]:
                        print("✗ Frame quality too low - improve lighting/focus")
                        continue
                    
                    # Get object info from user
                    print("\n" + "="*60)
                    print("CAPTURE REFERENCE IMAGE")
                    print("="*60)
                    
                    object_name = input("Enter object name: ").strip()
                    if not object_name:
                        print("✗ No name entered - skipping")
                        continue
                    
                    try:
                        width_mm = float(input("Enter object width (mm): "))
                        height_mm = float(input("Enter object height (mm): "))
                    except ValueError:
                        print("✗ Invalid dimensions - skipping")
                        continue
                    
                    self.save_reference(frame, object_name, (width_mm, height_mm))
                
                elif key == ord('f'):  # Toggle freeze
                    frozen = not frozen
                    if frozen:
                        frozen_frame = frame.copy()
                        print("✓ Frame frozen - review and press SPACE to capture")
                    else:
                        print("✓ Frame unfrozen")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()


def main():
    camera_index = 1
    capturer = ReferenceImageCapture(camera_index)
    capturer.run()


if __name__ == "__main__":
    main()