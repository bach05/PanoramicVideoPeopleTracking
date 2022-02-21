#!/bin/bash
gnome-terminal --window -- roslaunch tracking_demo monocular_person_following.launch
sleep 5
gnome-terminal --window -- roslaunch tracking_demo start_robot.launch webcam:=true publish_dummy_frames:=true camera_xyz:="0 0 1.074" camera_rpy:="0 0 0"
sleep 5
rviz -d monocular_person_following.rviz






