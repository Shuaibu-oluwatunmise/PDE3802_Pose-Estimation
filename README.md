# Object Detection and Pose Estimation with ROS TF

## Overview
This project implements real-time object detection and 6DOF pose estimation for 5 objects using a Raspberry Pi camera. The system detects objects, estimates their poses, and publishes their coordinates as TF frames relative to the camera frame for visualization in RViz.

## Objectives
- Detect and estimate pose for 5 different objects:
  - 2 objects with ArUco markers (full 6DOF pose)
  - 3 objects without markers (keypoint-based pose estimation)
- Publish object coordinates as TF frames in ROS2
- Visualize all frames in RViz

## System Architecture
```
Raspberry Pi Camera → Object Detection → Pose Estimation → TF Publisher → RViz
```

## Technical Approach
- **ArUco-tagged objects**: `cv2.aruco.detectMarkers()` + `cv2.solvePnP()` for precise pose
- **Non-tagged objects**: YOLOv8 keypoint detection + `cv2.solvePnP()` for full 6DOF pose
- **Camera calibration**: Intrinsic parameters obtained via checkerboard calibration
- **TF broadcasting**: Real-time publication of object transforms relative to camera frame

## Objects Detected
1. [Object 1 - ArUco tagged]
2. [Object 2 - ArUco tagged]
3. [Object 3 - Keypoint-based]
4. [Object 4 - Keypoint-based]
5. [Object 5 - Keypoint-based]

## Dependencies
- ROS2 Jazzy
- OpenCV 4.x with ArUco module
- Ultralytics YOLOv8
- tf2_ros
- numpy

## Limitations and Challenges
[To be completed after implementation]

---
**Course**: AI in Robotics (PDE3802)  
**Institution**: Middlesex University London  
**Date**: December 2025