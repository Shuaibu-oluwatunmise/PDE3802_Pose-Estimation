#!/usr/bin/env python3
"""
Launch file for Pose Estimation with RViz visualization
Launches:
- compromise_model3.py (pose estimation node) - HIGH PRIORITY
- RViz2 with custom config - DELAYED START, LOW PRIORITY
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the package root directory (one level up from launch/)
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(launch_dir)
    
    # Path to RViz config
    rviz_config = os.path.join(pkg_dir, 'config', 'pose_estimation.rviz')
    
    # Path to the Python script
    script_path = os.path.join(pkg_dir, 'main2.py')
    
    return LaunchDescription([
        # Launch the pose estimation node with HIGH PRIORITY
        ExecuteProcess(
            cmd=['nice', '-n', '-10', 'python3', script_path],
            output='log',  # Changed from 'screen' to reduce overhead
            name='pose_estimation_node'
        ),
        
        # Launch RViz2 after 8 seconds with LOW PRIORITY
        TimerAction(
            period=8.0,  # Wait 8 seconds before starting RViz
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config],
                    output='log',  # Changed from 'screen' to reduce overhead
                    prefix=['nice', '-n', '10']  # Low priority
                )
            ]
        )
    ])