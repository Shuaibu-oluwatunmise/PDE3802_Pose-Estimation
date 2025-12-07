import cv2
import numpy as np

CALIB_FILE = "camera_coeffs.npz"
REF_FILE = "reference_features4.npz"

# Pose estimation parameters
MIN_MATCH_COUNT   = 20
MIN_PNP_POINTS    = 8
MIN_PNP_INLIERS   = 10

UNDISTORT = False
DRAW_MATCH_LINES = True
DRAW_INLIERS = True

# -------------------------
# Stabilizer toggles
# -------------------------
USE_REPROJ_REJECTION   = True   # Reject frames with high reprojection error
USE_NORMAL_LOCK        = False   # Enforce consistent Z-axis orientation
USE_TRANSLATION_GUARD  = False   # Reject sudden large translation jumpse
USE_INLIER_REFINEMENT  = True   # Keep only low-error inliers for PnP

# Thresholds
REPROJ_ERROR_THRESH    = 3.0    # pixels
TRANSLATION_JUMP_THRESH = 0.05  # meters (5 cm)

from Camera_Calibration.camera_calibration.camera_params import CAMERA_MATRIX, DIST_COEFFS

def reconstruct_keypoints(kp_xy, kp_meta):
    keypoints = []
    for (x, y), (size, angle, resp, octv, cid) in zip(kp_xy, kp_meta):
        kp = cv2.KeyPoint(float(x), float(y), float(size),
                          float(angle), float(resp), int(octv), int(cid))
        keypoints.append(kp)
    return keypoints


def draw_keypoints(display, pts, color):
    for (x, y) in pts:
        cv2.circle(display, (int(x), int(y)), 4, color, -1)


def draw_match_lines(display, ref_kp, live_kp, matches):
    for m in matches:
        u1, v1 = ref_kp[m.queryIdx].pt
        u2, v2 = live_kp[m.trainIdx].pt
        cv2.line(display, (int(u1), int(v1)), (int(u2), int(v2)),
                 (255, 255, 0), 1)


def main():
    try:
        K = np.array(CAMERA_MATRIX, dtype=np.float32)
        dist = np.array(DIST_COEFFS, dtype=np.float32) if DIST_COEFFS is not None else None
        print("Camera calibration loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load calibration: {e}")
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        dist = None

    # Load reference data
    data = np.load(REF_FILE)
    ref_des = data["ref_des"]
    ref_kp_xy = data["ref_kp_xy"]
    ref_kp_meta = data["ref_kp_meta"]
    plane_ref = data["plane_ref"]
    book_width = float(data["book_width"])
    book_height = float(data["book_height"])
    ref_cam_w = int(data["cam_width"])
    ref_cam_h = int(data["cam_height"])

    # Rebuild keypoints
    ref_kp = reconstruct_keypoints(ref_kp_xy, ref_kp_meta)

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    # Check resolution consistency (Strict Mode)
    ret, test_frame = cap.read()
    if not ret:
        print("Camera error on first frame.")
        cap.release()
        return

    cam_h, cam_w = test_frame.shape[:2]
    if cam_w != ref_cam_w or cam_h != ref_cam_h:
        print(f"Resolution mismatch:")
        print(f"  Reference: {ref_cam_w}x{ref_cam_h}")
        print(f"  Current:   {cam_w}x{cam_h}")
        print("Please re-run reference_capture.py with this resolution.")
        cap.release()
        return

    prev_rvec = None
    prev_tvec = None
    prev_Rmat = None

    print("Tracking started. Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if UNDISTORT and dist is not None:
            frame = cv2.undistort(frame, K, dist)

        display = frame.copy()
        h, w = frame.shape[:2]

        # Detect features in live frame
        kp_live, des_live = orb.detectAndCompute(frame, None)

        if des_live is None:
            cv2.putText(display, "No features.", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Pose Tracking", display)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        # KNN matching + Lowe ratio test
        matches = bf.knnMatch(ref_des, des_live, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if DRAW_MATCH_LINES:
            draw_match_lines(display, ref_kp, kp_live, good)

        if len(good) < MIN_MATCH_COUNT:
            cv2.putText(display, f"Few matches ({len(good)})", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Pose Tracking", display)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        # Build 3D-2D correspondences
        obj_pts = []
        img_pts = []

        for m in good:
            P3 = plane_ref[m.queryIdx]
            if np.any(np.isnan(P3)):
                continue
            obj_pts.append(P3)
            img_pts.append(kp_live[m.trainIdx].pt)

        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)

        if len(obj_pts) < MIN_PNP_POINTS:
            cv2.putText(display, "Few PnP points.", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Pose Tracking", display)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        # PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K, dist,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if (not success) or inliers is None or len(inliers) < MIN_PNP_INLIERS:
            cv2.putText(display, "PnP failed.", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Pose Tracking", display)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        inliers = inliers.reshape(-1)
        obj_in = obj_pts[inliers]
        img_in = img_pts[inliers]

        # Compute reprojection errors
        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        errors = np.linalg.norm(proj - img_in, axis=1)
        mean_err = float(np.mean(errors))

        if DRAW_INLIERS:
            draw_keypoints(display, img_in, (255, 0, 0))  # blue

        # Optional inlier refinement
        if USE_INLIER_REFINEMENT:
            keep_idx = np.where(errors < REPROJ_ERROR_THRESH)[0]
            if len(keep_idx) >= MIN_PNP_POINTS:
                obj_ref = obj_in[keep_idx]
                img_ref = img_in[keep_idx]
                try:
                    success_ref, rvec_ref, tvec_ref = cv2.solvePnP(
                        obj_ref, img_ref, K, dist,
                        rvec, tvec, True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    if success_ref:
                        rvec, tvec = rvec_ref, tvec_ref
                        obj_in, img_in = obj_ref, img_ref
                        proj, _ = cv2.projectPoints(obj_in, rvec, tvec, K, dist)
                        proj = proj.reshape(-1, 2)
                        errors = np.linalg.norm(proj - img_in, axis=1)
                        mean_err = float(np.mean(errors))
                except cv2.error:
                    pass

        # Reprojection-based frame rejection
        if USE_REPROJ_REJECTION and mean_err > REPROJ_ERROR_THRESH:
            if prev_rvec is not None and prev_tvec is not None:
                rvec = prev_rvec.copy()
                tvec = prev_tvec.copy()
                status = f"Frame rejected (err={mean_err:.2f})"
            else:
                status = "High error, no previous pose."
                cv2.putText(display, status, (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.imshow("Pose Tracking", display)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue
        else:
            status = f"Reproj err={mean_err:.2f}px"

        # Normal direction lock
        if USE_NORMAL_LOCK:
            Rmat, _ = cv2.Rodrigues(rvec)
            normal = Rmat[:, 2]
            if normal[2] < 0:
                Rmat[:, 2] *= -1
                tvec[2] *= -1
                rvec, _ = cv2.Rodrigues(Rmat)
        else:
            Rmat, _ = cv2.Rodrigues(rvec)

        # Translation jump guard
        if prev_tvec is not None and USE_TRANSLATION_GUARD:
            jump = np.linalg.norm(tvec - prev_tvec)
            if jump > TRANSLATION_JUMP_THRESH:
                rvec = prev_rvec.copy()
                tvec = prev_tvec.copy()
                status = f"Jump rejected ({jump:.2f}m)"

        # Accept pose
        prev_rvec = rvec.copy()
        prev_tvec = tvec.copy()
        prev_Rmat = Rmat.copy()

        # Draw axis
        axis_len = min(book_width, book_height) * 0.5
        axis_3d = np.float32([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len]
        ])

        imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
        imgpts = imgpts.reshape(-1, 2)

        origin = tuple(imgpts[0].astype(int))
        x_end = tuple(imgpts[1].astype(int))
        y_end = tuple(imgpts[2].astype(int))
        z_end = tuple(imgpts[3].astype(int))

        cv2.line(display, origin, x_end, (0, 0, 255), 3)
        cv2.line(display, origin, y_end, (0, 255, 0), 3)
        cv2.line(display, origin, z_end, (255, 0, 0), 3)

        # Pose text
        t = tvec.ravel()
        cv2.putText(display, f"X:{t[0]:.3f}m", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(display, f"Y:{t[1]:.3f}m", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(display, f"Z:{t[2]:.3f}m", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(display, status, (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Pose Tracking", display)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
