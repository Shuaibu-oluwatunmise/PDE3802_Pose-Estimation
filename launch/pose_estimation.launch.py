#!/usr/bin/env python3
"""
Launch file for Pose Estimation with RViz visualization
Launches:
- compromise_model3.py (pose estimation node)
- RViz2 with custom config
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package root directory (one level up from launch/)
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(launch_dir)
    
    # Path to RViz config
    rviz_config = os.path.join(pkg_dir, 'config', 'pose_estimation.rviz')
    
    # Path to the Python script
    script_path = os.path.join(pkg_dir, 'compromise_model3.py')
    
    return LaunchDescription([
        # Launch the pose estimation node
        ExecuteProcess(
            cmd=['python3', script_path],
            output='screen',
            name='pose_estimation_node'
        ),
        
        # Launch RViz2 with config
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen'
        )
    ])
