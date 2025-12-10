# Object Detection and Pose Estimation with ROS2 TF

Real-time 6DOF pose estimation system for multiple objects using Raspberry Pi camera. Detects objects, estimates their poses, and publishes coordinates as ROS2 TF frames for visualization in RViz.

[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](https://opencv.org/)

---

## Table of Contents

- [System Overview](#system-overview)
- [Prerequisites](#prerequisites)
- [SSH Connection Setup](#ssh-connection-setup)
- [Installation](#installation)
- [Running the System](#running-the-system)
- [Daily Usage](#daily-usage)
- [Visualization in RViz](#visualization-in-rviz)
- [Repository Structure](#repository-structure)
- [Technical Approach](#technical-approach)
- [Controls](#controls)
- [Notes & Limitations](#notes--limitations)
- [Authors](#authors)

---

## System Overview

### Detection & Tracking Methods

- **3 objects** tracked via homography-based pose estimation (keypoint matching)
- **2 objects** tracked via ArUco markers (fiducial-based pose)

### Output

- 6DOF pose estimation (3D position + 3D orientation)
- Real-time TF frame broadcasting to `/tf` topic
- Visual feedback with 3D coordinate axes overlay

---

## Prerequisites

### Hardware
- Raspberry Pi (tested on Pi 4)
- Raspberry Pi Camera Module (CSI interface)
- Laptop/PC for SSH connection

### Software
- Ubuntu 24.04 (or compatible) on Raspberry Pi
- ROS2 Jazzy
- Python 3.10+

### Network
- Both Raspberry Pi and laptop must be connected to the same network

---

## SSH Connection Setup

### Step 1: Get Raspberry Pi IP Address

On the Raspberry Pi, run:
```bash
hostname -I
```

Note the IP address (e.g., `192.168.1.100`)

### Step 2: Connect from Your Laptop

On your laptop, run:
```bash
ssh ros@<ip-address>
```

Replace `<ip-address>` with the IP address from Step 1.

Enter the password when prompted.

---

## Installation

### 1. Clone the Repository
```bash
cd ~
git clone https://github.com/Shuaibu-oluwatunmise/PDE3802_Pose-Estimation.git
```

### 2. Run Automated Setup Script
```bash
cd ~/PDE3802_Pose-Estimation/Set_Up
bash setup_workspace.sh
```

**What the script does:**
- Creates workspace at `~/pose_estimation_ws_t1/`
- Moves repository into `~/pose_estimation_ws_t1/src/`
- Creates Python virtual environment (`pose_est`) with system site packages
- Installs all dependencies from `requirements.txt`
- Creates activation script (`initialise_environment.sh`)
- Adds `init_env` alias to your `.bashrc`

### 3. Reload Shell and Activate Environment
```bash
source ~/.bashrc
init_env
```

### 4. Activate Environment on Raspberry Pi

If you're running via SSH, also run on the Raspberry Pi terminal (desktop/direct access):
```bash
init_env
```

---

## Running the System

### Step 1: Camera Calibration (First Time Only)
```bash
init_env
cd ~/pose_estimation_ws_t1/src/PDE3802_Pose-Estimation/Camera_Calibration
python3 Camera_Calibration.py
```

Follow the on-screen instructions to calibrate your camera using a checkerboard pattern.

### Step 2: Create Required `__init__.py` File
```bash
cd ~/pose_estimation_ws_t1/src/PDE3802_Pose-Estimation/Camera_Calibration/camera_calibration
touch __init__.py
```

### Step 3: Return to Repository Root
```bash
cd ~/pose_estimation_ws_t1/src/PDE3802_Pose-Estimation
```

### Step 4: Run Main Pose Estimation System
```bash
python3 main2.py
```

---

## Daily Usage

Every time you open a new terminal:
```bash
init_env
```

**This command:**
- ✅ Sources ROS2 Jazzy
- ✅ Activates Python venv (`pose_est`)
- ✅ Sets X11 display permissions (for `cv2.imshow` over SSH)
- ✅ Sources workspace overlay (if built)

---

## Visualization in RViz

Open a new terminal and run:
```bash
init_env
rviz2
```

**In RViz:**
1. Add **TF** display
2. Set **Fixed Frame** to `camera_link_G4`
3. View object frames: `[object_name]_frame`

---

## Repository Structure
```
PDE3802_Pose-Estimation/
├── pose_estimator_ros2_v2.py         # Main ROS2 node (PRIMARY FILE)
├── combined_pose_estimation.py       # Standalone version (no ROS2)
├── homography_pose_webcam.py         # Development/testing version
│
├── Camera_Calibration/
│   ├── Camera_Calibration.py         # Calibration script
│   └── camera_calibration/
│       ├── camera_params.py          # Intrinsic parameters
│       └── images/                   # Calibration images
│
├── Data_Preparation/
│   ├── video_data_collection.py     # Record calibration videos
│   ├── extract_frames.py            # Extract frames from videos
│   └── auto_annotation.py           # Automated labeling pipeline
│
├── Detection/
│   └── train.py                      # YOLO training script
│
├── runs/detect/yolov8n_detect_V1/
│   ├── weights/best.pt              # Trained model weights
│   └── [training metrics]           # Loss curves, confusion matrix
│
└── Set_Up/
    ├── setup_workspace.sh           # Automated installation
    ├── requirements.txt             # Python dependencies
    └── SETUP_README.md              # Detailed setup guide
```

---

## Technical Approach

### Detection

- **YOLOv8n** object detector trained on ~14,000 annotated images
- Average of ~2,700 images per object class
- Custom dataset with varied lighting, angles, and backgrounds

### Pose Estimation

#### Homography-based (3 objects):

1. YOLO detects object bounding box
2. ORB feature extraction within ROI
3. Feature matching with reference image
4. Homography matrix calculation (RANSAC)
5. PnP solver estimates 6DOF pose

#### ArUco-based (2 objects):

1. YOLO detects object bounding box
2. ArUco marker detection within ROI
3. Direct PnP solver for 6DOF pose (IPPE_SQUARE)

### Camera Calibration

- Checkerboard-based intrinsic calibration
- Parameters stored in `camera_params.py`
- Used for accurate pose estimation via `cv2.solvePnP()`

---

## Controls

| Key   | Action                        |
|-------|-------------------------------|
| `r`   | Reset homography calibrations |
| `s`   | Save current frame            |
| `ESC` | Exit                          |

---

## Notes & Limitations

### Notes

- **Object selection flexibility:** The current model (v1) is trained on 5 specific objects, but the system can work with any objects by retraining the detection model and updating object dimensions in the configuration.
- **Homography requirements:** Objects must have sufficient texture/features for keypoint matching.
- **ArUco visibility:** Markers must remain visible and unoccluded for pose estimation.

### Dependencies

- **ROS2 Jazzy**
- **Python 3.10+**
- **OpenCV 4.x** with ArUco module
- **Ultralytics YOLOv8**
- **tf2_ros**
- **NumPy**
- **GStreamer** (for Raspberry Pi camera)

---

## Authors

- **Oluwatunmise Raphael Shuaibu** - M00960413
- **Myu Wai Shin** - M00964135
- **Arash Bazrafshan** - M00766882

**Course:** AI in Robotics (PDE3802)  
**Institution:** Middlesex University London  
**Date:** December 2024

---

## License

This project is part of an academic assessment and is not licensed for commercial use.

---

## Acknowledgments

Special thanks to the course instructors and teaching staff at Middlesex University London for their guidance and support throughout this project.