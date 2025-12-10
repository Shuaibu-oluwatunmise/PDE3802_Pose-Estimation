#!/usr/bin/env python3
"""
Launch file for Pose Estimation with RViz visualization
RViz launches with BOTH delay AND after tracker confirmation
Auto-restart on crash
"""

import os
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, ExecuteProcess, TimerAction
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package root directory
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(launch_dir)
    
    # Path to RViz config
    rviz_config = os.path.join(pkg_dir, 'config', 'pose_estimation.rviz')
    
    # Path to the Python script
    script_path = os.path.join(pkg_dir, 'main.py')
    
    # Define the tracker node with respawn
    tracker_node = ExecuteProcess(
        cmd=['python3', script_path],
        output='screen',
        name='pose_estimation_node',
        respawn=True,  # Auto-restart if crashes
        respawn_delay=2.0  # Wait 2 seconds before restart
    )
    
    # RViz with BOTH conditions: wait for tracker start AND delay
    rviz_delayed = RegisterEventHandler(
        OnProcessStart(
            target_action=tracker_node,
            on_start=[
                TimerAction(
                    period=12.0,  # Wait 12 seconds AFTER tracker starts
                    actions=[
                        Node(
                            package='rviz2',
                            executable='rviz2',
                            name='rviz2',
                            arguments=['-d', rviz_config],
                            output='log'
                        )
                    ]
                )
            ]
        )
    )
    
    # Log when tracker exits
    tracker_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=tracker_node,
            on_exit=lambda event, context: print(f"\n[ERROR] Tracker exited with code {event.returncode}\n")
        )
    )
    
    return LaunchDescription([
        tracker_node,
        rviz_delayed,
        tracker_exit_handler
    ])