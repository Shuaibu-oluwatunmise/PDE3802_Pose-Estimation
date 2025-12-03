#!/bin/bash
# =============================================================================
# Pose Estimation Workspace - Setup Script
# =============================================================================
# Run this from: ~/PDE3802_Pose-Estimation/setup/
# This script will:
#   1. Create workspace structure
#   2. Move this repo into workspace/src/
#   3. Create Python venv
#   4. Install dependencies
#   5. Create init_env command
# =============================================================================

set -e  # Exit on any error

WORKSPACE_NAME="pose_estimation_ws_t1"
VENV_NAME="pose_est"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$CURRENT_DIR")"
REPO_NAME="$(basename "$REPO_ROOT")"

echo ""
echo "============================================================"
echo "POSE ESTIMATION WORKSPACE - SETUP"
echo "============================================================"
echo ""
echo "Current location: $CURRENT_DIR"
echo "Repository root: $REPO_ROOT"
echo "Repository name: $REPO_NAME"
echo ""
echo "This will:"
echo "  1. Create workspace: ~/$WORKSPACE_NAME"
echo "  2. Move '$REPO_NAME' into ~/$WORKSPACE_NAME/src/"
echo "  3. Create Python venv: $VENV_NAME"
echo "  4. Install dependencies from requirements.txt"
echo "  5. Create 'init_env' command"
echo ""
read -p "Press ENTER to continue or Ctrl+C to cancel..."

# =============================================================================
# 1. Check Prerequisites
# =============================================================================
echo ""
echo "🔍 Checking prerequisites..."

# Check if ROS2 Jazzy exists
if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    echo "❌ ERROR: ROS2 Jazzy not found!"
    echo "Please install ROS2 Jazzy first."
    exit 1
fi
echo "✓ ROS2 Jazzy found"

# Check if we're in the right directory
if [ ! -f "$CURRENT_DIR/requirements.txt" ]; then
    echo "❌ ERROR: requirements.txt not found in $CURRENT_DIR"
    echo "Please run this script from the setup/ folder"
    exit 1
fi
echo "✓ requirements.txt found"

# =============================================================================
# 2. Create Workspace Structure
# =============================================================================
echo ""
echo "📁 Creating workspace structure..."

# Create workspace if it doesn't exist
if [ -d ~/$WORKSPACE_NAME ]; then
    echo "⚠ Workspace ~/$WORKSPACE_NAME already exists!"
    read -p "Do you want to continue? This will overwrite files. (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

mkdir -p ~/$WORKSPACE_NAME/src
echo "✓ Workspace created: ~/$WORKSPACE_NAME"

# =============================================================================
# 3. Move Repository into Workspace
# =============================================================================
echo ""
echo "📦 Moving repository into workspace..."

# Check if destination already exists
if [ -d ~/$WORKSPACE_NAME/src/$REPO_NAME ]; then
    echo "⚠ Repository already exists in workspace, removing old version..."
    rm -rf ~/$WORKSPACE_NAME/src/$REPO_NAME
fi

# Move the entire repository
mv "$REPO_ROOT" ~/$WORKSPACE_NAME/src/
echo "✓ Repository moved to: ~/$WORKSPACE_NAME/src/$REPO_NAME"

# Update paths since we moved
WORKSPACE_DIR=~/$WORKSPACE_NAME
REPO_DIR=$WORKSPACE_DIR/src/$REPO_NAME
SETUP_DIR=$REPO_DIR/Set_Up

# =============================================================================
# 4. Create Python Virtual Environment
# =============================================================================
echo ""
echo "🐍 Creating Python virtual environment..."

cd $WORKSPACE_DIR

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Create venv with system site packages (for ROS2)
python3 -m venv $VENV_NAME --system-site-packages
source $VENV_NAME/bin/activate
echo "✓ Virtual environment created: $VENV_NAME"

# Upgrade pip
pip install --upgrade pip
echo "✓ pip upgraded"

# =============================================================================
# 5. Install Python Dependencies
# =============================================================================
echo ""
echo "📚 Installing Python dependencies..."

pip install -r $SETUP_DIR/requirements.txt
echo "✓ Dependencies installed"

# Verify installations
echo ""
echo "Verifying installations:"
pip list | grep ultralytics && echo "  ✓ ultralytics" || echo "  ⚠ ultralytics not found"
pip list | grep opencv && echo "  ✓ opencv-python" || echo "  ⚠ opencv-python not found"
pip list | grep numpy && echo "  ✓ numpy" || echo "  ⚠ numpy not found"
pip list | grep transforms3d && echo "  ✓ transforms3d" || echo "  ⚠ transforms3d not found"

# =============================================================================
# 6. Create Initialization Script
# =============================================================================
echo ""
echo "📝 Creating initialization script..."

cat > $WORKSPACE_DIR/initialise_environment.sh << 'EOF'
#!/bin/bash
# =============================================================================
# Pose Estimation Environment Initialization
# =============================================================================
# Activates ROS2 Jazzy + Python venv + workspace overlay
# Run this in every new terminal or use: init_env
# =============================================================================

# 🟢 Authorize local user to draw on the display (for cv2.imshow over SSH)
if [ -n "$DISPLAY" ]; then
    xhost +local:$(whoami) > /dev/null 2>&1
    echo "✓ X11 local permissions set for user '$(whoami)'"
fi

# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# Activate Python venv
source ~/pose_estimation_ws_t1/pose_est/bin/activate

# Source workspace overlay (after building ROS2 packages)
if [ -f ~/pose_estimation_ws_t1/install/setup.bash ]; then
    source ~/pose_estimation_ws_t1/install/setup.bash
fi

echo "✓ ROS2 Jazzy sourced"
echo "✓ pose_est venv activated"
echo "✓ Workspace ready: ~/pose_estimation_ws_t1"
echo ""
echo "Python: $(which python3)"
EOF

chmod +x $WORKSPACE_DIR/initialise_environment.sh
echo "✓ Created: $WORKSPACE_DIR/initialise_environment.sh"

# =============================================================================
# 7. Create Alias in .bashrc
# =============================================================================
echo ""
echo "🔗 Creating 'init_env' alias..."

# Check if alias already exists
if grep -q "alias init_env=" ~/.bashrc; then
    echo "⚠ Alias 'init_env' already exists in ~/.bashrc, skipping..."
else
    echo "" >> ~/.bashrc
    echo "# Pose Estimation Workspace Initialization" >> ~/.bashrc
    echo "alias init_env='source ~/pose_estimation_ws_t1/initialise_environment.sh'" >> ~/.bashrc
    echo "✓ Alias added to ~/.bashrc"
fi

# =============================================================================
# 8. Final Summary
# =============================================================================
echo ""
echo "============================================================"
echo "✓ SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Workspace created at: ~/$WORKSPACE_NAME"
echo "Repository location: ~/$WORKSPACE_NAME/src/$REPO_NAME"
echo "Python venv: $VENV_NAME"
echo ""
echo "📌 IMPORTANT: Reload your shell configuration:"
echo "   source ~/.bashrc"
echo ""
echo "📌 To activate the environment in any terminal:"
echo "   init_env"
echo ""
echo "📁 Workspace structure:"
echo "   ~/$WORKSPACE_NAME/"
echo "   ├── $VENV_NAME/                    # Python venv"
echo "   ├── initialise_environment.sh      # Activation script"
echo "   └── src/"
echo "       └── $REPO_NAME/"
echo "           ├── src/Detection/         # Main code"
echo "           └── setup/                 # Setup files"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: source ~/.bashrc"
echo "   2. Run: init_env"
echo "   3. Navigate: cd ~/pose_estimation_ws_t1/src/$REPO_NAME/src/Detection"
echo "   4. Test camera calibration or pose tracking"
echo ""
echo "============================================================"