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


def humans_to_msg(humans):
    persons = Persons()

    for human in humans:
        person = Person()

        for k in human.body_parts:
            body_part = human.body_parts[k]

            body_part_msg = BodyPartElm()
            body_part_msg.part_id = body_part.part_idx
            body_part_msg.x = body_part.x
            body_part_msg.y = body_part.y
            body_part_msg.confidence = body_part.score
            person.body_part.append(body_part_msg)
        persons.persons.append(person)

    return persons


def callback_image(data):
    # et = time.time()
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(data, "bgr8")

        IM_H = cv_image.shape[0]
        IM_W = cv_image.shape[1]

        tileW = IM_W/3
        overlap = int(tileW * 0.06)

        imageC = cv_image[int(IM_H*0.2):int(IM_H*0.9),tileW+tileW-overlap:]

        CROP_H = imageC.shape[0]
        CROP_W = imageC.shape[1]
        
    except CvBridgeError as e:
        rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
        return

    acquired = tf_lock.acquire(False)
    if not acquired:
        return

    try:

        # Inference on imageC
        humansC = pose_estimator1.inference(imageC, resize_to_default=True, upsample_size=resize_out_ratio)

        # Fix detections
        OFF_X = tileW+tileW-overlap
        OFF_Y = int(IM_H*0.2)
        CROP_RATIO = (CROP_W*1.0 / IM_W, 1.0*CROP_H / IM_H)
        humans = TfPoseEstimator.cropped2full_humans(humansC, IM_W, IM_H, CROP_RATIO, OFF_X, OFF_Y)

    finally:
        tf_lock.release()

    msg = humans_to_msg(humans)
    msg.image_w = 960
    msg.image_h = 480
    msg.header = data.header

    pub_pose.publish(msg)

if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('TfPoseEstimatorROS', anonymous=True, log_level=rospy.INFO)

    # parameters
    image_topic = rospy.get_param('~camera', '')
    model = rospy.get_param('~model', 'cmu')

    resolution1 = rospy.get_param('~resolution1', '640x640')
    pub_topic = rospy.get_param('~pub_topic', '~poseC')
    resize_out_ratio = float(rospy.get_param('~resize_out_ratio', '4.0'))
    tf_lock = Lock()

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided.')
        sys.exit(-1)

    try:
        w1, h1 = model_wh(resolution1)
        graph_path = get_graph_path(model)

        rospack = rospkg.RosPack()
        graph_path = os.path.join(rospack.get_path('tfpose_ros'), graph_path)
    except Exception as e:
        rospy.logerr('invalid model: %s, e=%s' % (model, e))
        sys.exit(-1)

    pose_estimator1 = TfPoseEstimator(graph_path, target_size=(w1, h1))
    cv_bridge = CvBridge()

    rospy.Subscriber(image_topic, Image, callback_image, queue_size=4000, buff_size=2**24)
    pub_pose = rospy.Publisher(pub_topic, Persons, queue_size=4000)

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
