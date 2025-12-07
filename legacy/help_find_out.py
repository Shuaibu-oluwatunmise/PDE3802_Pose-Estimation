import cv2
import numpy as np

CALIB_FILE = "camera_coeffs.npz"
OUTPUT_FILE = "reference_features4.npz"

# Real-world dimensions of the book (meters)q

# Electro-Board
#BOOK_WIDTH_M  = 0.12
#BOOK_HEIGHT_M = 0.17

# KIT
#BOOK_WIDTH_M  = 0.19
#BOOK_HEIGHT_M = 0.25

# BOOK
BOOK_WIDTH_M  = 0.074
BOOK_HEIGHT_M = 0.048

MIN_FEATURES = 20
UNDISTORT = False


from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

def serialize_keypoints(keypoints):
    """Convert cv2.KeyPoint list into arrays we can store in npz."""
    kp_xy = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    kp_meta = np.array([
        (kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in keypoints
    ], dtype=np.float32)
    return kp_xy, kp_meta

def main():
    try:
        K = np.array(CAMERA_MATRIX, dtype=np.float32)
        dist = np.array(DIST_COEFFS, dtype=np.float32) if DIST_COEFFS is not None else None
        print("Camera calibration loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        dist = None

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    orb = cv2.ORB_create(nfeatures=2000)

    rect_tl = rect_br = None
    sx = sy = None

    print("Reference capture mode.")
    print("Place the book flat inside the rectangle.")
    print("Press SPACE to capture, 'q' to quit.\n")

    ref_img = None
    ref_kp = None
    ref_des = None
    plane_ref = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cam_h, cam_w = frame.shape[:2]

        if UNDISTORT and dist is not None:
            frame = cv2.undistort(frame, K, dist)

        display = frame.copy()

        # Build rectangle once based on frame size and book aspect
        if rect_tl is None:
            aspect = BOOK_WIDTH_M / BOOK_HEIGHT_M
            rect_h = int(cam_h * 0.7)
            rect_w = int(rect_h * aspect)

            if rect_w > int(cam_w * 0.7):
                rect_w = int(cam_w * 0.7)
                rect_h = int(rect_w / aspect)

            cx, cy = cam_w // 2, cam_h // 2
            rect_tl = (cx - rect_w // 2, cy - rect_h // 2)
            rect_br = (cx + rect_w // 2, cy + rect_h // 2)

            sx = BOOK_WIDTH_M / rect_w
            sy = BOOK_HEIGHT_M / rect_h

        # Draw rectangle and instructions
        cv2.rectangle(display, rect_tl, rect_br, (0, 255, 0), 2)
        cv2.putText(display, "Align book fully in rectangle, flat.",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.putText(display, "Press SPACE to capture reference.",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
        cv2.putText(display, "Press 'q' to quit.",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)

        cv2.imshow("Reference Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Reference capture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == 32:  # SPACE
            ref_img = frame.copy()
            ref_kp, ref_des = orb.detectAndCompute(ref_img, None)

            if ref_des is None or len(ref_kp) < MIN_FEATURES:
                print(f"Too few features ({0 if ref_des is None else len(ref_kp)}). Try again.")
                continue

            # Build 3D plane coords for each keypoint
            plane_ref = []
            rx, ry = rect_tl
            rect_w = rect_br[0] - rect_tl[0]
            rect_h = rect_br[1] - rect_tl[1]

            valid = 0
            for kp in ref_kp:
                u, v = kp.pt
                x_px = u - rx
                y_px = v - ry

                if 0 <= x_px <= rect_w and 0 <= y_px <= rect_h:
                    X = x_px * sx
                    Y = -y_px * sy
                    plane_ref.append([X, Y, 0.0])
                    valid += 1
                else:
                    plane_ref.append([np.nan, np.nan, np.nan])

            print(f"Captured {len(ref_kp)} keypoints, {valid} on-plane.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if ref_img is None or ref_des is None:
        print("No valid reference captured. Exiting.")
        return

    # Serialize keypoints and prepare arrays
    ref_kp_xy, ref_kp_meta = serialize_keypoints(ref_kp)
    plane_ref = np.array(plane_ref, dtype=np.float32)

    # Save everything needed for tracking
    np.savez_compressed(
        OUTPUT_FILE,
        ref_img=ref_img,
        ref_des=ref_des,
        ref_kp_xy=ref_kp_xy,
        ref_kp_meta=ref_kp_meta,
        plane_ref=plane_ref,
        rect_tl=np.array(rect_tl, dtype=np.int32),
        rect_br=np.array(rect_br, dtype=np.int32),
        sx=np.float32(sx),
        sy=np.float32(sy),
        book_width=BOOK_WIDTH_M,
        book_height=BOOK_HEIGHT_M,
        cam_width=np.int32(cam_w),
        cam_height=np.int32(cam_h)
    )

    print(f"\nReference saved to '{OUTPUT_FILE}'.")
    print("You can now run pose_tracking.py using this reference.")


if __name__ == "__main__":
    main()
