#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from monocular_people_tracking.msg import *
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import pandas as pd
import math

import message_filters

rows = []

def callback(tracks, tracks_corrected, gt):

    if(len(tracks.tracks) == 0): return

    for i in range(len(tracks.tracks)):

        track = tracks.tracks[i]
        track_a = tracks_corrected.tracks[i]
        
        target_id = track.id
        
        frame = gt.header.seq

        pos_x = track.pos.x
        pos_y = track.pos.y
        pos_z = track.height

        pos_x_a = track_a.pos.x
        pos_y_a = track_a.pos.y
        pos_z_a = track_a.height

        gt_pos_x = gt.tracks[0].pos.x
        gt_pos_y = gt.tracks[0].pos.y
        gt_pos_z = gt.tracks[0].height

        gt_neck_x = gt.tracks[0].associated_neck_ankle[0].x
        gt_neck_y = gt.tracks[0].associated_neck_ankle[0].y
        gt_ankle_x = gt.tracks[0].associated_neck_ankle[1].x
        gt_ankle_y = gt.tracks[0].associated_neck_ankle[1].y

        if(len(track.associated_neck_ankle)) == 0:
            neck_x = -1
            neck_y = -1
            ankle_x = -1
            ankle_y = -1
        else:
            neck_x = track.associated_neck_ankle[0].x
            neck_y = track.associated_neck_ankle[0].y

        if(len(track.associated_neck_ankle)) <= 1:
            ankle_x = -1
            ankle_y = -1
        else:
            ankle_x = track.associated_neck_ankle[1].x
            ankle_y = track.associated_neck_ankle[1].y

        if(ankle_x > 2000 or ankle_y > 2000):
            ankle_y = -1
            ankle_x = -1

        rows.append([frame, pos_x, pos_y, pos_z, neck_x, neck_y, ankle_x, ankle_y])
        result = '%d,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%f,%f,%f,%d,%d,%d,%d,%d' % (frame, pos_x, pos_y, pos_z, pos_x_a, pos_y_a, pos_z_a, neck_x, neck_y, ankle_x, ankle_y, gt_pos_x, gt_pos_y, gt_pos_z, gt_neck_x,gt_neck_y, gt_ankle_x, gt_ankle_y,target_id)
        
        print(result)
    


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    #image_sub = message_filters.Subscriber('/theta_camera/qhd/image_raw/compressed', CompressedImage)
    tracks_sub = message_filters.Subscriber('/monocular_people_tracking/tracks', TrackArray)
    tracks_corrected_sub = message_filters.Subscriber('/monocular_people_tracking/tracks_corrected', TrackArray)
    gt_sub = message_filters.Subscriber('/gt', TrackArray)

    #ts = message_filters.TimeSynchronizer([image_sub, tracks_sub, tracks_corrected_sub, gt_sub], 100)
    ts = message_filters.TimeSynchronizer([tracks_sub, tracks_corrected_sub, gt_sub], 100)
    ts.registerCallback(callback)


    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    print('frame,pos_x,pos_y,pos_z,pos_x_a,pos_y_a,pos_z_a,neck_x,neck_y,ankle_x,ankle_y,gt_pos_x,gt_pos_y,gt_pos_z,gt_neck_x,gt_neck_y,gt_ankle_x,gt_ankle_y,target_id')
    listener()
