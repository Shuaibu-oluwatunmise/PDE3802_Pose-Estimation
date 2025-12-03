import cv2
import numpy as np                

def rotation_to_euler(rvec):                
    rmat, _ = cv2.Rodrigues(rvec)                
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)                
    if sy > 1e-6:                
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])                
        pitch = np.arctan2(-rmat[2, 0], sy)                
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])                
    else:                
        roll = np.arctan2(-rmat[1, 2], rmat[1, 1])                
        pitch = np.arctan2(-rmat[2, 0], sy)                
        yaw = 0                
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)                

def draw_3d_axes(frame, rvec, tvec, K, D, color, length=100):                
    # ... (identical draw logic) ...                
    return frame                

def setup_display():                
    import os                
    if "DISPLAY" not in os.environ:                
        print("WARN: Defaulting to physical display :0")                
        os.environ["DISPLAY"] = ":0"                
    if "XAUTHORITY" not in os.environ:                
        if os.path.exists("/home/ros/.Xauthority"):                
            os.environ["XAUTHORITY"] = "/home/ros/.Xauthority"