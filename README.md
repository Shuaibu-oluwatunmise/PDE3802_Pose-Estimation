Object Detection and Pose Estimation with ROS2 TF
Real-time 6DOF pose estimation system for multiple objects using Raspberry Pi camera. Detects objects, estimates their poses, and publishes coordinates as ROS2 TF frames for visualization in RViz.

System Overview
Detection & Tracking Methods:

3 objects tracked via homography-based pose estimation (keypoint matching)
2 objects tracked via ArUco markers (fiducial-based pose)
Output:

6DOF pose estimation (3D position + 3D orientation)
Real-time TF frame broadcasting to /tf topic
Visual feedback with 3D coordinate axes overlay
Quick Start
1. Installation (Raspberry Pi)
cd ~
git clone https://github.com/Shuaibu-oluwatunmise/PDE3802_Pose-Estimation.git
cd PDE3802_Pose-Estimation/Set_Up
bash setup_workspace.sh
Reload shell and activate environment:

source ~/.bashrc
init_env
2. Run Main System
cd ~/pose_estimation_ws_t1/src/PDE3802_Pose-Estimation
python3 pose_estimator_ros2.py
3. Visualize in RViz
Open new terminal:

init_env
ros2 run rviz2 rviz2
In RViz:

Add TF display
Set fixed frame to camera_link
View object frames: [object_name]_frame
Controls
Key	Action
r	Reset homography calibrations
s	Save current frame
ESC	Exit
Technical Approach
Detection
YOLOv8n object detector trained on ~14,000 annotated images
Average of ~2,700 images per object class
Custom dataset with varied lighting, angles, and backgrounds
Pose Estimation
Homography-based (3 objects):

YOLO detects object bounding box
ORB feature extraction within ROI
Feature matching with reference image
Homography matrix calculation (RANSAC)
PnP solver estimates 6DOF pose
ArUco-based (2 objects):

YOLO detects object bounding box
ArUco marker detection within ROI
Direct PnP solver for 6DOF pose (IPPE_SQUARE)
Camera Calibration
Checkerboard-based intrinsic calibration
Parameters stored in camera_params.py
Used for accurate pose estimation via cv2.solvePnP()
Repository Structure
PDE3802_Pose-Estimation/
├── pose_estimator_ros2.py          # Main ROS2 node (PRIMARY FILE)
├── combined_pose_estimation.py     # Standalone version (no ROS2)
├── homography_pose_webcam.py       # Development/testing version
│
├── Camera_Calibration/
│   ├── Camera_Calibration.py       # Calibration script
│   └── camera_calibration/
│       ├── camera_params.py        # Intrinsic parameters
│       └── images/                 # Calibration images
│
├── Data_Preparation/
│   ├── video_data_collection.py   # Record calibration videos
│   ├── extract_frames.py          # Extract frames from videos
│   └── auto_annotation.py         # Automated labeling pipeline
│
├── Detection/
│   └── train.py                    # YOLO training script
│
├── runs/detect/yolov8n_detect_V1/
│   ├── weights/best.pt            # Trained model weights
│   └── [training metrics]         # Loss curves, confusion matrix
│
└── Set_Up/
    ├── setup_workspace.sh         # Automated installation
    ├── requirements.txt           # Python dependencies
    └── SETUP_README.md            # Detailed setup guide
Dependencies
ROS2 Jazzy
Python 3.10+
OpenCV 4.x with ArUco module
Ultralytics YOLOv8
tf2_ros
NumPy
GStreamer (for Raspberry Pi camera)
Notes
Object selection flexibility: The current model (v1) is trained on 5 specific objects, but the system is designed to work with any objects by retraining the detection model and updating object dimensions in the configuration.
Homography requirements: Objects must have sufficient texture/features for keypoint matching.
ArUco visibility: Markers must remain visible and unoccluded for pose estimation.
Dataset
[Training Results]
To be updated with training metrics, loss curves, and confusion matrices

[Model Evaluation and Error Analysis]
To be updated with validation results and error analysis

[Test Results]
To be updated with real-world test scenarios and accuracy measurements

Limitations and Challenges
To be completed after full system evaluation

Authors
Oluwatunmise Raphael Shuaibu M00960413
Myu Wai Shin M00964135
Arash Bazrafshan M00766882
Course: AI in Robotics (PDE3802)
Institution: Middlesex University London
Date: December 2024