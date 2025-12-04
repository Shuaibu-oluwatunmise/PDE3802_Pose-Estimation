"""
CSI Camera Calibration - Raspberry Pi
Working version with camera_params.py output
"""

import os
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["GST_PLUGIN_PATH"] = "/usr/local/lib/aarch64-linux-gnu/gstreamer-1.0:" + os.environ.get("GST_PLUGIN_PATH", "")

if "DISPLAY" not in os.environ:
    print("WARN: No DISPLAY variable found. Defaulting to physical display :0")
    os.environ["DISPLAY"] = ":0"

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import numpy as np
import cv2
import json
from datetime import datetime

# === Parameters ===
chessboard_size = (9, 6)       # inner corners (cols, rows)
square_size = 24.0             # mm
max_frames = 20                # number of good views to collect

# === Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) scaled by square_size
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []   # 3D points in world
imgpoints = []   # 2D points in image
img_size = None  # will be set from first frame

# Create output directories
os.makedirs('camera_calibration/images', exist_ok=True)

init_path = 'camera_calibration/__init__.py'
if not os.path.exists(init_path):
    with open(init_path, 'w') as f:
        pass  # Just create an empty file
    print(f"✓ Created: {init_path}")

# === Init GStreamer / libcamera pipeline ===
Gst.init(None)

pipeline = Gst.parse_launch(
    "libcamerasrc ! "
    "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink name=sink emit-signals=true max-buffers=2 drop=true"
)
sink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

def pull_frame(timeout_ns=10_000_000):
    """Grab one BGR frame from appsink, or None if timed out."""
    sample = sink.emit("try-pull-sample", timeout_ns)
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
        return frame
    finally:
        buf.unmap(mapinfo)

print("\n" + "="*60)
print("RASPBERRY PI CSI CAMERA CALIBRATION")
print("="*60)
print(f"\n📋 Checkerboard: {chessboard_size[0]}x{chessboard_size[1]} interior corners")
print(f"📏 Square size: {square_size}mm")
print(f"📸 Images to capture: {max_frames}")
print("\nInstructions:")
print("  - Hold checkerboard in different positions")
print("  - Press SPACE when GREEN corners appear")
print("  - Press ESC to quit")
print("="*60 + "\n")

cv2.namedWindow("Calibration - CSI", cv2.WINDOW_NORMAL)

try:
    while True:
        frame = pull_frame()
        if frame is None:
            # no frame yet; let GUI update
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]  # (width, height)

        # Detect corners
        found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, chessboard_size, corners, found)
            msg = f'Corners found - SPACE to save ({len(objpoints)}/{max_frames})'
            cv2.putText(display, msg, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, 'Show chessboard to camera...',
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        cv2.imshow("Calibration - CSI", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("[INFO] Exiting without full calibration.")
            break
        elif key == 32 and found:  # SPACE
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            
            # Save image
            img_path = f'camera_calibration/images/calib_{len(objpoints)-1:02d}.jpg'
            cv2.imwrite(img_path, frame)
            
            print(f"[INFO] Frame {len(objpoints)}/{max_frames} captured.")
            if len(objpoints) >= max_frames:
                print("[INFO] Collected required frames.")
                break
finally:
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()

# === Calibration ===
if len(objpoints) >= 5 and img_size is not None:
    print("\n" + "="*60)
    print("PERFORMING CALIBRATION")
    print("="*60)
    print(f"Using {len(objpoints)} images...")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints_reproj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
        mean_error += error
    mean_error /= len(objpoints)
    
    print("\n" + "="*60)
    print("✓ CALIBRATION SUCCESSFUL!")
    print("="*60)
    print(f"\nReprojection Error: {mean_error:.4f} pixels")
    print(f"\nCamera Matrix:")
    print(camera_matrix)
    print(f"\nDistortion Coefficients:")
    print(dist_coeffs.ravel())
    print("="*60)
    
    # Save to NPZ (original format)
    np.savez("rpi_csi_calibration.npz",
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs)
    print("\n✓ Saved: rpi_csi_calibration.npz")
    
    # Save to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f'camera_calibration/camera_calibration_{timestamp}.json'
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'rvecs': [r.tolist() for r in rvecs],
        'tvecs': [t.tolist() for t in tvecs],
        'reprojection_error': mean_error,
        'checkerboard_size': chessboard_size,
        'square_size': square_size,
        'resolution': '640x480',
        'camera_type': 'Raspberry Pi CSI',
        'timestamp': timestamp
    }
    
    with open(json_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    print(f"✓ Saved: {json_path}")
    
    # Save to Python file (for easy import)
    py_path = 'camera_calibration/camera_params.py'
    with open(py_path, 'w') as f:
        f.write('"""Camera calibration parameters for Raspberry Pi CSI Camera"""\n')
        f.write('import numpy as np\n\n')
        f.write(f'# Calibrated on {timestamp}\n')
        f.write(f'# Resolution: 640x480\n')
        f.write(f'# Reprojection error: {mean_error:.4f} pixels\n')
        f.write(f'# Camera: Raspberry Pi CSI with libcamera\n\n')
        f.write('CAMERA_MATRIX = np.array([\n')
        for row in camera_matrix:
            f.write(f'    {row.tolist()},\n')
        f.write('], dtype=np.float32)\n\n')
        f.write(f'DIST_COEFFS = np.array({dist_coeffs.ravel().tolist()}, dtype=np.float32)\n')
    
    print(f"✓ Saved: {py_path}")
    print("\n" + "="*60)
    print("✓ CALIBRATION COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("  - rpi_csi_calibration.npz")
    print(f"  - {json_path}")
    print(f"  - {py_path}")
    print("\nNext: Run pose tracking with calibrated camera!")
    
else:
    print("[WARN] Not enough valid frames collected. Calibration aborted.")