#!/usr/bin/env python
import time
import os
import sys
import ast
import numpy as np
import cv2

from threading import Lock
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from tfpose_ros.msg import Persons, Person, BodyPartElm

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import model_wh, get_graph_path

from monocular_person_following.msg import *

def callback_image(data):
    # et = time.time()
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")
        cv_image = np.concatenate((cv_image[:, cv_image.shape[1]-107:], cv_image, cv_image[:, 0:107]), axis=1)
        #cv2.imshow('image', cv_image)
        #cv2.waitKey(10)
    except CvBridgeError as e:
        rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
        return

    extended_image = cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
    extended_image.header = data.header

    pub_image.publish(extended_image)

if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('TfPoseEstimatorROS', anonymous=True, log_level=rospy.INFO)

    # parameters
    image_topic = '/theta_camera/image_raw'#rospy.get_param('~camera', '')
    model = rospy.get_param('~model', 'cmu')

    pub_topic = rospy.get_param('~pub_topic', '~image_raw')

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided.')
        sys.exit(-1)

    cv_bridge = CvBridge()

    rospy.Subscriber(image_topic, Image, callback_image, queue_size=10, buff_size=2**24)
    pub_image = rospy.Publisher(pub_topic, Image, queue_size=5)

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
