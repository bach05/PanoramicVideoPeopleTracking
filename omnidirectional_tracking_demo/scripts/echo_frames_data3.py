#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from monocular_people_tracking.msg import *
from monocular_person_following.msg import *
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError
import pandas as pd
import math
from tfpose_ros.msg import Persons, Person, BodyPartElm
import message_filters

parts_names = {0: 'nose',
               1: 'neck',
               2: 'rshoulder',
               3: 'relbow',
               4: 'rwrist',
               5: 'lshoulder',
               6: 'lelbow',
               7: 'lwrist',
               8: 'rhip',
               9: 'rknee',
               10: 'rankle',
               11: 'lhip',
               12: 'lknee',
               13: 'lankle',
               14: 'reye',
               15: 'leye',
               16: 'rear',
               17: 'lear',
               18: 'background'
               }

def msg_to_humans(msg):
    humans = []
    for person in msg.persons:
        human = {}
        for body_part in person.body_part:
            human[body_part.part_id] = [body_part.x, body_part.y,
                                        body_part.confidence, parts_names[body_part.part_id]]
        humans.append(human)
    return humans


def callback(target, tracks, tracks_corrected, gt):

    frame = gt.header.seq
    target_id = target.target_id
    state_name = target.state.data

    if(target_id < 0): return

    for i in range(len(tracks.tracks)):
        track = tracks.tracks[i]
        track_a = tracks_corrected.tracks[i]

        if(track.id != target_id):
            continue

        gt_pos_x = gt.tracks[0].pos.x
        gt_pos_y = gt.tracks[0].pos.y
        gt_pos_z = gt.tracks[0].height

        gt_pos_x_a = gt.tracks[1].pos.x
        gt_pos_y_a = gt.tracks[1].pos.y
        gt_pos_z_a = gt.tracks[1].height

        pos_x = track.pos.x
        pos_y = track.pos.y
        pos_z = track.height

        pos_x_a = track_a.pos.x
        pos_y_a = track_a.pos.y
        pos_z_a = track_a.height

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

        detected = len(tracks.tracks)
        result = '%d,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%d,%d,%d,%d,%d,%d' % (frame, pos_x, pos_y, pos_z, pos_x_a, pos_y_a, pos_z_a, neck_x, neck_y, ankle_x, ankle_y, gt_pos_x, gt_pos_y, gt_pos_z, gt_pos_x_a, gt_pos_y_a, gt_pos_z_a, gt_neck_x,gt_neck_y, gt_ankle_x, gt_ankle_y,target_id, detected)
        print(result)

        global file
        f = open(file, "a")
        f.write(result+'\n')
        f.close()

def callback1(pose1, pose2, pose):

    frame = pose.header.seq

    p1 = len(pose1.persons)
    p2 = len(pose2.persons)
    p = len(pose.persons)

    gd1 = min(p1,1)*1.0
    gd2 = min(p2,1)*1.0
    gd = min(p,1)*1.0

    fd1 = p1-gd1
    fd2 = p2-gd2
    fd = p-gd

    M1_1 = gd1/ (gd1+fd1) if gd1+fd1>0 else 0
    M1_2 = gd2/ (gd2+fd2) if gd2+fd2>0 else 0
    M1 = gd/ (gd+fd) if gd+fd>0 else 0

    result = '%d,%d,%d,%d'%(frame,p1,p2,p)

    global file1
    f1 = open(file1, "a")
    f1.write(result+'\n')
    f1.close()

def callback2(pose):

    frame = pose.header.seq

    p1 = 0
    p2 = 0
    p = len(pose.persons)

    gd1 = min(p1,1)*1.0
    gd2 = min(p2,1)*1.0
    gd = min(p,1)*1.0

    fd1 = p1-gd1
    fd2 = p2-gd2
    fd = p-gd

    M1_1 = gd1/ (gd1+fd1) if gd1+fd1>0 else 0
    M1_2 = gd2/ (gd2+fd2) if gd2+fd2>0 else 0
    M1 = gd/ (gd+fd) if gd+fd>0 else 0

    result = '%d,%d,%d,%d'%(frame,p1,p2,p)

    global file1
    f1 = open(file1, "a")
    f1.write(result+'\n')
    f1.close()

def callback3(pose, gt):

    frame = gt.header.seq

    # Middle point from ground truth
    gt_neck_x = gt.tracks[0].associated_neck_ankle[0].x
    gt_neck_y = gt.tracks[0].associated_neck_ankle[0].y
    gt_ankle_x = gt.tracks[0].associated_neck_ankle[1].x
    gt_ankle_y = gt.tracks[0].associated_neck_ankle[1].y
    gt_mean_x = (gt_neck_x + gt_ankle_x) * 0.5
    gt_mean_y = (gt_neck_y + gt_ankle_y) * 0.5

    # Middle point from detection
    det_mean_x = -1
    det_mean_y = -1
    det_neck_x = -1
    det_neck_y = -1
    det_ankle_x = -1
    det_ankle_y = -1

    min_dist = 99999999
    min_man_dist = -1
    

    for person in pose.persons:
        neck_x = -1
        neck_y = -1
        lankle_x = -1
        lankle_y = -1
        rankle_x = -1
        rankle_y = -1
        ankle_x = -1
        ankle_y = -1
        neck = False
        lankle = False
        rankle = False
        for bp in person.body_part:
            if(bp.part_id == 1): # neck
                neck_x = bp.x * 1920.0
                neck_y = bp.y * 960.0
                neck = True
            elif(bp.part_id == 13): # lankle
                lankle_x = bp.x * 1920.0
                lankle_y = bp.y * 960.0
                lankle = True
            elif(bp.part_id == 10): # rankle
                rankle_x = bp.x * 1920.0
                rankle_y = bp.y * 960.0
                rankle = True
            
            if(neck and (lankle or rankle)):
                if(lankle and rankle):
                    ankle_x = (lankle_x + rankle_x) * 0.5
                    ankle_y = (lankle_y + rankle_y) * 0.5
                elif(lankle):
                    ankle_x = lankle_x
                    ankle_y = lankle_y
                elif(rankle):
                    ankle_x = rankle_x
                    ankle_y = rankle_y

                mean_x = (neck_x + ankle_x) * 0.5
                mean_y = (neck_y + ankle_y) * 0.5

                diff_x = min(abs(mean_x - gt_mean_x), 1920 - abs(mean_x - gt_mean_x))

                dist = (diff_x)**2 + (mean_y - gt_mean_y)**2
                man_dist = abs(diff_x) + abs(mean_y - gt_mean_y)

                if(dist < min_dist):
                    det_mean_x = mean_x
                    det_mean_y = mean_y
                    det_neck_x = neck_x
                    det_neck_y = neck_y
                    det_ankle_x = ankle_x
                    det_ankle_y = ankle_y
                    min_dist = dist
                    min_man_dist = man_dist
                    

    gt_pos_x_a = gt.tracks[1].pos.x
    gt_pos_y_a = gt.tracks[1].pos.y
    cam_dist = math.sqrt(gt_pos_x_a**2+gt_pos_y_a**2)

    min_dist = math.sqrt(min_dist) if min_dist < 99999 else -1
    result = '%d,%3.2f,%d,%d,%4.2f,%4.2f,%4.2f,%4.2f,%d,%d,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f,%4.2f'%(frame,cam_dist,gt_neck_x,gt_neck_y,gt_ankle_x,gt_ankle_y,gt_mean_x,gt_mean_y,det_neck_x,det_neck_y,det_ankle_x,det_ankle_y,det_mean_x,det_mean_y,min_dist,min_man_dist)

    global file2
    f2 = open(file2, "a")
    f2.write(result+'\n')
    f2.close()

    


def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener2', anonymous=True)

    #image_sub = message_filters.Subscriber('/theta_camera/qhd/image_raw/compressed', CompressedImage)
    target_sub = message_filters.Subscriber('/monocular_person_following/target', Target)
    tracks_sub = message_filters.Subscriber('/monocular_people_tracking/tracks', TrackArray)
    tracks_corrected_sub = message_filters.Subscriber('/monocular_people_tracking/tracks_corrected', TrackArray)
    gt_sub = message_filters.Subscriber('/gt', TrackArray)


    pose_topic = '/pose_estimator/pose'
    pose1_topic = '/pose1_estimator/pose1'
    pose2_topic = '/pose2_estimator/pose2'

    # subscribers
    pose_sub = message_filters.Subscriber(pose_topic, Persons)
    pose1_sub = message_filters.Subscriber(pose1_topic, Persons)
    pose2_sub = message_filters.Subscriber(pose2_topic, Persons)

    global file
    file = rospy.get_param('~name', "/home/filippo/dataset/result/test_result.csv")
    global file1
    file1 = file+'_detection.csv'
    global file2
    file2 = file+'_det_M2.csv'


    #ts = message_filters.TimeSynchronizer([image_sub, tracks_sub, tracks_corrected_sub, gt_sub], 100)
    ts = message_filters.TimeSynchronizer([target_sub,tracks_sub,tracks_corrected_sub,gt_sub], 100)
    ts.registerCallback(callback)

    #ts1 = message_filters.TimeSynchronizer([pose1_sub,pose2_sub,pose_sub], 100)
    ts1 = message_filters.TimeSynchronizer([pose_sub], 100)
    ts1.registerCallback(callback2)

    ts2 = message_filters.TimeSynchronizer([pose_sub, gt_sub], 100)
    ts2.registerCallback(callback3)

    columns = 'frame,pos_x,pos_y,pos_z,pos_x_a,pos_y_a,pos_z_a,neck_x,neck_y,ankle_x,ankle_y,gt_pos_x,gt_pos_y,gt_pos_z,gt_pos_x_a,gt_pos_y_a,gt_pos_z_a,gt_neck_x,gt_neck_y,gt_ankle_x,gt_ankle_y,target_id,detected'
    print(columns)
    
    f = open(file, "w")
    f.write(columns+'\n')
    f.close()

    f1 = open(file1, "w")
    f1.write('frame,pose1,pose2,pose'+'\n')
    f1.close()

    f2 = open(file2, "w")
    f2.write('frame,cam_dist,gt_neck_x,gt_neck_y,gt_ankle_x,gt_ankle_y,gt_mean_x,gt_mean_y,det_neck_x,det_neck_y,det_ankle_x,det_ankle_y,det_mean_x,det_mean_y,euc_dist,man_dist'+'\n')
    f2.close()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
