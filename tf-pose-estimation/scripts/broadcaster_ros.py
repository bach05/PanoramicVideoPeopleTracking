#!/usr/bin/env python
import time
import os
import sys
import ast

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
    except CvBridgeError as e:
        rospy.logerr('[tf-pose-estimation] Converting Image Error. ' + str(e))
        return

    acquired = tf_lock.acquire(False)
    if not acquired:
        return

    try:

        # Inference on the full image by default
        humans = pose_estimator1.inference(cv_image, resize_to_default=True, upsample_size=resize_out_ratio)
        
        global USE_CROPPED
        if(USE_CROPPED):

            IM_H = cv_image.shape[0]
            IM_W = cv_image.shape[1]

            FOV_X = 36.0
            FOV_RATIO = 1.0 #368.0 / 432.0
            FOV_Y = FOV_X * FOV_RATIO
            
            
            FOV_W = FOV_X * IM_W / 360
            FOV_H = FOV_Y * IM_H / 180

            CROP_RATIO = (FOV_W / IM_W, FOV_H / IM_H)
            CROP_SIZE = (int(IM_W * CROP_RATIO[0]), int(IM_H * CROP_RATIO[1]))

            assert( CROP_SIZE[0] <= IM_W and CROP_SIZE[1] <= IM_H )
            
            global CROP_CENTER
            
            CC = (CROP_CENTER[0], CROP_CENTER[1])

            from_x = CC[0]-CROP_SIZE[0]/2
            to_x = CC[0]+CROP_SIZE[0]/2

            if(from_x < 0):
                CC = (CROP_SIZE[0]/2, CC[1])

            if(to_x > IM_W):
                CC = (IM_W - CROP_SIZE[0]/2, CC[1]) 

            cropped = cv_image[CC[1]-CROP_SIZE[1]/2:CC[1]+CROP_SIZE[1]/2, CC[0]-CROP_SIZE[0]/2:CC[0]+CROP_SIZE[0]/2].copy()

            rospy.loginfo('original image (%4d, %4d) -> cropped image (%4d, %4d)' % (IM_W, IM_H, cropped.shape[1], cropped.shape[0]))
            rospy.loginfo('crop center (%4d, %4d)' % (CC[0], CC[1]))

            humans2 = pose_estimator2.inference(cropped, resize_to_default=True, upsample_size=resize_out_ratio)
            
            OFF_X = CC[0]-CROP_SIZE[0]/2
            OFF_Y = CC[1]-CROP_SIZE[1]/2

            humans2 = TfPoseEstimator.cropped2full_humans(humans2, IM_W, IM_H, CROP_RATIO, OFF_X, OFF_Y)
            
            humans = TfPoseEstimator.associate_humans(humans, humans2, IM_W, IM_H, CROP_RATIO, OFF_X, OFF_Y)

    finally:
        tf_lock.release()

    msg = humans_to_msg(humans)
    msg.image_w = data.width
    msg.image_h = data.height
    msg.header = data.header

    if(USE_CROPPED) : 
        msg.crop_w = CROP_SIZE[0]
        msg.crop_h = CROP_SIZE[1]
        msg.crop_x = CC[0]-CROP_SIZE[0]/2
        msg.crop_y = CC[1]-CROP_SIZE[1]/2

    pub_pose.publish(msg)

def target_callback(target_msg):

    global RE_ID_COUNTER
    global USE_CROPPED
    use_cropped = USE_CROPPED
    global CROP_CENTER
    state_name = target_msg.state.data
    target_id = target_msg.target_id
    rospy.loginfo(state_name)

    distance = target_msg.distance
    rospy.loginfo('distance : %2.2f' % (distance))

    if(state_name == 'tracking' and distance > 2.0):
        use_cropped = True
        RE_ID_COUNTER = 0
        if(target_msg.center_of_mass.z > 0):
            # Update cropped image position based on the center of mass
            CROP_CENTER = (int(target_msg.center_of_mass.x), CROP_CENTER[1])
            
    elif state_name =='re-identification':
        rospy.loginfo('RE_ID_COUNTER : %3d' % (RE_ID_COUNTER))
        use_cropped = True
        if(RE_ID_COUNTER >= 5):
            CROP_CENTER = ((CROP_CENTER[0]+144)%1920, CROP_CENTER[1])
            RE_ID_COUNTER = 0
            
        else:
            RE_ID_COUNTER += 1
    else:
        use_cropped = False

    if(target_msg.center_of_mass.z < 0):
            # Turn off cropped image
            use_cropped = False

    USE_CROPPED = True# use_cropped
        

if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('TfPoseEstimatorROS', anonymous=True, log_level=rospy.INFO)

    global USE_CROPPED
    USE_CROPPED = False
    global UPDATE_CROPPED
    UPDATE_CROPPED = False
    global CROP_CENTER
    CROP_CENTER = (960, 515)

    global RE_ID_COUNTER
    RE_ID_COUNTER = 0

    # parameters
    image_topic = rospy.get_param('~camera', '')
    model = rospy.get_param('~model', 'cmu')

    resolution1 = rospy.get_param('~resolution1', '640x320')
    resolution2 = rospy.get_param('~resolution2', '192x192')
    resize_out_ratio = float(rospy.get_param('~resize_out_ratio', '4.0'))
    tf_lock = Lock()

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided.')
        sys.exit(-1)

    try:
        w1, h1 = model_wh(resolution1)
        w2, h2 = model_wh(resolution2)
        graph_path = get_graph_path(model)

        rospack = rospkg.RosPack()
        graph_path = os.path.join(rospack.get_path('tfpose_ros'), graph_path)
    except Exception as e:
        rospy.logerr('invalid model: %s, e=%s' % (model, e))
        sys.exit(-1)

    pose_estimator1 = TfPoseEstimator(graph_path, target_size=(w1, h1))
    pose_estimator2 = TfPoseEstimator(graph_path, target_size=(w2, h2))
    cv_bridge = CvBridge()

    rospy.Subscriber(image_topic, Image, callback_image, queue_size=1, buff_size=2**24)
    target_sub = rospy.Subscriber('/monocular_person_following/target', Target, target_callback, queue_size=1)
    pub_pose = rospy.Publisher('~pose', Persons, queue_size=1)

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
