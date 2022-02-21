#!/usr/bin/env python
import time
import os
import sys
import ast
import numpy as np
import cv2
import pandas as pd

from threading import Lock
import rospy
import rospkg
from bondpy import bondpy
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import *

from monocular_person_following.msg import *
from monocular_people_tracking.msg import *

global wait_next
wait_next = True

def callback(data):
    wait_next = False
    rospy.loginfo('trigger')
    return


if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('GroundTruthCamera', anonymous=True, log_level=rospy.INFO)

        

    # Receives id from A using a service or an action
    bond = bondpy.Bond("example_bond_topic", '10')
    bond.start()

    # parameters
    image_topic = '/theta_camera/image_raw'
    gt_topic = '/gt'

    cv_bridge = CvBridge()

    pub_image = rospy.Publisher(image_topic, Image, queue_size=1) #500

    pub_gt = rospy.Publisher(gt_topic, TrackArray, queue_size=3)

    video_path = rospy.get_param('~video', "/home/filippo/dataset/video_set/video0.mp4")

    video_name = video_path.split('/')[-1].split('.')[0]
    gt_pos = '/home/filippo/dataset/pos/gt_'+video_name+'_pos.csv'

    rospy.Subscriber('/next_frame', std_msgs.msg.String, callback, queue_size=1)

    #tf_lock = Lock()

    theta = cv2.VideoCapture(video_path)

    # Read gt dataframe
    df = pd.read_csv(gt_pos)
    name = video_name+'_frame'

    ret, frame = theta.read()

    #cv2.imshow('frame', frame)
    #cv2.waitKey(1)


    count = 0

    while(ret):
        now = time.time()            # get the time

        msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8")
        msg_frame.header.frame_id = 'theta_camera_optical_frame'
        msg_frame.header.stamp = rospy.Time.now()
        msg_frame.header.seq = count
        # Create gt msg
        candidate = df.loc[df['frame'] == name+str(count)]
        
        if(len(candidate) > 0):

            gt_frame = candidate.iloc[0]
            print(gt_frame)

            tracks = TrackArray()

            track0 = Track()
            track0.pos.x = gt_frame.pos_x
            track0.pos.y = gt_frame.pos_y
            track0.pos.z = 0
            track0.height = gt_frame.pos_z
            neck = Point32()
            neck.x = gt_frame.x_image_neck
            neck.y = gt_frame.y_image_neck
            ankle = Point32()
            ankle.x = gt_frame.x_image_ankle
            ankle.y = gt_frame.y_image_ankle
            track0.associated_neck_ankle = [neck, ankle]

            track1 = Track()
            track1.pos.x = gt_frame.pos_x_a
            track1.pos.y = gt_frame.pos_y_a
            track1.pos.z = 0
            track1.height = gt_frame.pos_z_a
            track1.associated_neck_ankle = [neck, ankle]
            tracks.header = msg_frame.header
            tracks.tracks = [track0, track1]

            # Publish msgs
            
            pub_gt.publish(tracks)
            pub_image.publish(msg_frame)
            rospy.wait_for_message("/next_frame", std_msgs.msg.String)
        else:
            if(count%4 == 0):
                pub_image.publish(msg_frame)
                rospy.wait_for_message("/next_frame", std_msgs.msg.String)
        
        # Read new frame
        ret, frame = theta.read()
        # Display the resulting frame
        if(not ret):
            #cv2.waitKey(1)
            break


        # Display the resulting frame
        #cv2.imshow('frame', frame)
        #cv2.waitKey(1)
        count += 1 

        elapsed = time.time() - now  # how long was it running?
        #time.sleep(max(0,1./10.-elapsed))       # sleep accordingly so the full iteration takes 1 second
        wait_next = True

        

        #while(wait_next):
        #    rospy.loginfo('waiting')
        #    pass


    bond.break_bond()

    rospy.loginfo('finished')
