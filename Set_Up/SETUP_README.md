# Pose Estimation Workspace - Setup Guide

## Quick Setup (Automated)

### 1. Clone the repository:
```bash
cd ~
git clone https://github.com/Shuaibu-oluwatunmise/PDE3802_Pose-Estimation.git
```

### 2. Run the setup script:
```bash
cd ~/PDE3802_Pose-Estimation/Set_Up
bash setup_workspace.sh
```

### 3. Reload shell and activate:
```bash
source ~/.bashrc
init_env
```

---

## What the Setup Script Does

1. **Creates workspace** at `~/pose_estimation_ws_t1/`
2. **Moves this repository** into `~/pose_estimation_ws_t1/src/`
3. **Creates Python venv** named `pose_est` with system site packages
4. **Installs dependencies** from `requirements.txt`
5. **Creates activation script** (`initialise_environment.sh`)
6. **Adds `init_env` alias** to your `.bashrc`

---

## Final Structure

```
~/pose_estimation_ws_t1/
├── pose_est/                          # Python virtual environment
├── initialise_environment.sh          # Environment activation script
├── src/
│   └── PDE3802_Pose-Estimation/        # This repository (moved here)
│       ├── src/
│       │   └── Detection/
│       │       ├── Csi_camera_calibration.py
│       │       ├── camera_calibration/
│       │       └── ... (your code)
│       └── setup/
│           ├── setup_workspace.sh
│           ├── requirements.txt
│           └── SETUP_README.md (this file)
└── install/ build/ log/               # Created when building ROS2 packages
```

---

## Daily Usage

Every time you open a new terminal:

```bash
init_env
```

This command:
- ✅ Sources ROS2 Jazzy
- ✅ Activates Python venv (`pose_est`)
- ✅ Sets X11 display permissions (for cv2.imshow over SSH)
- ✅ Sources workspace overlay (if built)

---

## Common Commands

```bash
# Activate environment
init_env

# Navigate to Detection folder
cd ~/pose_estimation_ws_t1/src/PDE3802_Pose-Estimation/src/Detection

# Run camera calibration (on Pi desktop)
export QT_QPA_PLATFORM=wayland
python3 Csi_camera_calibration.py

# Run pose tracking
python3 pi_csi_pose_tracker.py

# Build ROS2 packages
cd ~/pose_estimation_ws_t1
colcon build

# Check Python location (should show venv path)
which python3
```

---

## Prerequisites

Before running setup, ensure you have:

- ✅ Raspberry Pi with Ubuntu/Raspberry Pi OS
- ✅ ROS2 Jazzy installed (`/opt/ros/jazzy/setup.bash` exists)
- ✅ Python 3 installed
- ✅ Git installed
- ✅ Internet connection

---

## Troubleshooting

### Setup script fails:
```bash
# Check ROS2 is installed
ls /opt/ros/jazzy/setup.bash

# Check Python version
python3 --version

# Check you're in the right directory
pwd  # Should show: ~/PDE3802_Pose-Estimation/setup
```

### `init_env` command not found:
```bash
# Reload bashrc
source ~/.bashrc

# Or manually source the script
source ~/pose_estimation_ws_t1/initialise_environment.sh
```

### Display issues when running CV scripts over SSH:
```bash
# Set display variable
export DISPLAY=:0
export QT_QPA_PLATFORM=wayland

# Or run directly on Pi desktop (not via SSH)
```

### Python not from venv:
```bash
init_env
which python3  # Should show: ~/pose_estimation_ws_t1/pose_est/bin/python3
```

### Dependencies not installing:
```bash
# Activate venv manually
source ~/pose_estimation_ws_t1/pose_est/bin/activate

# Install manually
pip install -r ~/pose_estimation_ws_t1/src/office-item-classifier/setup/requirements.txt
```

---

## Manual Setup (If Automated Fails)

If the automated script fails, you can set up manually:

```bash
# 1. Create workspace
mkdir -p ~/pose_estimation_ws_t1/src

# 2. Move repository
mv ~/PDE3802_Pose-Estimation ~/pose_estimation_ws_t1/src/

# 3. Create venv
cd ~/pose_estimation_ws_t1
source /opt/ros/jazzy/setup.bash
python3 -m venv pose_est --system-site-packages
source pose_est/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r src/PDE3802_Pose-Estimation/setup/requirements.txt

# 5. Create init script
cat > initialise_environment.sh << 'EOF'
#!/bin/bash
if [ -n "$DISPLAY" ]; then
    xhost +local:$(whoami) > /dev/null 2>&1
fi
source /opt/ros/jazzy/setup.bash
source ~/pose_estimation_ws_t1/pose_est/bin/activate
if [ -f ~/pose_estimation_ws_t1/install/setup.bash ]; then
    source ~/pose_estimation_ws_t1/install/setup.bash
fi
echo "✓ Environment ready"
EOF

chmod +x initialise_environment.sh

# 6. Add alias
echo "alias init_env='source ~/pose_estimation_ws_t1/initialise_environment.sh'" >> ~/.bashrc
source ~/.bashrc
```

---

## Getting Help

- Check the main README: `~/pose_estimation_ws_t1/src/PDE3802_Pose-Estimation/README.md`
- Review workspace structure: `ls -la ~/pose_estimation_ws_t1/`
- Check Python packages: `init_env && pip list`
- Verify ROS2: `init_env && ros2 --version`

---

**Setup Version:** 1.0  
**Last Updated:** December 2024