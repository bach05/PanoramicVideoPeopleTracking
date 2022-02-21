#!/usr/bin/env python
import time
import cv2
import math
import rospy
import numpy as np
import message_filters
import std_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from tfpose_ros.msg import Persons, Person, BodyPartElm
from tf_pose.estimator import Human, BodyPart, TfPoseEstimator

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

def getBoundingBox(person, parts, IM_W, IM_H):
    # generate bounding box
    minX = 9999
    maxX = 0
    minY = 9999
    maxY = 0

    for k in parts:

        body_part = person[k]
        minX = int(body_part[0]*IM_W) if minX > int(body_part[0]*IM_W) else minX
        maxX = int(body_part[0]*IM_W) if maxX < int(body_part[0]*IM_W) else maxX
        minY = int(body_part[1]*IM_H) if minY > int(body_part[1]*IM_H) else minY
        maxY = int(body_part[1]*IM_H) if maxY < int(body_part[1]*IM_H) else maxY 

    tl = (minX, minY)
    br = (maxX, maxY)

    return tl, br

def rectArea(tl, br):
    width = abs(br[0]-tl[0])
    height = abs(tl[1]-br[1])
    area = width * height
    if(area == 0):
        return 1
    else:
        return area

def interRect(tl1, br1, tl2, br2):
    x1 = max(tl1[0], tl2[0])
    y1 = max(tl1[1], tl2[1])

    x2 = min(br1[0], br2[0])
    y2 = min(br1[1], br2[1])

    return (x1, y1), (x2, y2)


def associate_humans(humans1, humans2, IM_W, IM_H, P_COUNT_THR=10, JAC_THR=0.35):

    parts = [1, 2, 5, 8, 11] # 

    new_humans = []
    
    # LOOP OVER PEOPLE DETECTED IN THE SMALL IMAGE
    for h2 in humans2:

        # IF H2 DOESN'T HAVE ENOUGH PARTS IT IS DISCARDED IN ADVANCE
        if len(h2) < P_COUNT_THR:
            continue

        # Check which joints in parts are in h2
        h2_parts = [i for i in parts if i in h2.keys()]

        # print("H2_PARTS : ",h2_parts)
        # IF LESS THAN 3 POINTS AMONG parts ARE IN H2 IT GETS DISCARDED IN ADVANCE
        if(len(h2_parts) < 4):
            continue

        # generate bounding box
        tl2, br2 = getBoundingBox(h2, h2_parts, IM_W, IM_H)

        area2 = rectArea(tl2, br2)
        rospy.loginfo('area h2 = %d' % area2)

        #img = cv2.rectangle(img, tl, br, (255,0,0), 2)
        #cv2.imshow('merger', img)
        #cv2.waitKey(1)


        jacc_list = []

        # LOOP OVER PEOPLE DETECTED IN THE SMALL IMAGE
        for h1 in humans1:

            # Check which joints in parts are in h1
            h1_parts = [i for i in parts if i in h1.keys()]

            # generate bounding box
            # generate bounding box
            tl1, br1 = getBoundingBox(h1, h1_parts, IM_W, IM_H)

            area1 = rectArea(tl1, br1)
            rospy.loginfo('area h1 = %d' % area1)


            tl, br = interRect(tl1, br1, tl2, br2)

            if(tl[0] <= br[0] and tl[1] <= br[1]):
                area = rectArea(tl, br)
            else:
                area = 0

            #iou = area*1.0 / (area1 + area2 - area)


            iou = (1.0* area ) / min(area1, area2)
            
            rospy.loginfo('area = %d' % area)
            rospy.loginfo('IoU = %f' % iou)

            
            # Jaccard
            jacc_list.append(iou)

        # print("Jaccard List : ",jacc_list)
        best_jacc = 0.0
        if(len(humans1)):
            best_jacc = max(jacc_list)

        if(len(humans1) and best_jacc > JAC_THR):
            max_index = jacc_list.index(best_jacc)  # max jaccard score
            # Merge the two readings
            for i in parts_names.keys():
                # if i in h1.body_parts.keys() and i in h2.body_parts.keys():
                if i in h2.keys():
                    # print("Merging : ", i)
                    humans1[max_index][i] = h2[i]
        else:
            # print("Adding H2")
            new_humans.append(h2)

    return humans1 + new_humans

def associate_humans2(humans1, humans2, IM_W, IM_H, CROP_RATIO, OFF_X, OFF_Y, P_COUNT_THR=10, DIST_THR=0.06, JAC_THR=0.5):

    parts = [1, 2, 5, 8, 11]

    new_humans = []
    
    img = np.zeros((IM_H,IM_W,3), dtype=np.uint8)

    # LOOP OVER PEOPLE DETECTED IN THE SMALL IMAGE
    for h2 in humans2:

        # IF H2 DOESN'T HAVE ENOUGH PARTS IT IS DISCARDED IN ADVANCE
        if len(h2) < P_COUNT_THR:
            continue

        # Check which joints in parts are in h2
        h2_parts = [i for i in parts if i in h2.keys()]

        # print("H2_PARTS : ",h2_parts)
        # IF LESS THAN 3 POINTS AMONG parts ARE IN H2 IT GETS DISCARDED IN ADVANCE
        if(len(h2_parts) < 4):
            continue

        # generate bounding box
        minX = 9999
        maxX = 0
        minY = 9999
        maxY = 0

        for k in h2_parts:

            body_part = h2[k]
            minX = int(body_part[0]*IM_W) if minX > int(body_part[0]*IM_W) else minX
            maxX = int(body_part[0]*IM_W) if maxX < int(body_part[0]*IM_W) else maxX
            minY = int(body_part[1]*IM_H) if minY > int(body_part[1]*IM_H) else minY
            maxY = int(body_part[1]*IM_H) if maxY < int(body_part[1]*IM_H) else maxY 

        tl = (minX, minY)
        br = (maxX, maxY)

        area = (maxY - minY) * (maxX - minX)
        rospy.loginfo('area h2 = %d' % area)

        #img = cv2.rectangle(img, tl, br, (255,0,0), 2)
        #cv2.imshow('merger', img)
        #cv2.waitKey(1)


        jacc_list = []

        # LOOP OVER PEOPLE DETECTED IN THE SMALL IMAGE
        for h1 in humans1:

            inter_size = 0
            union_size = 0

            # Check which joints in parts are in h1
            h1_parts = [i for i in parts if i in h1.keys()]

            # generate bounding box
            minX = 9999
            maxX = 0
            minY = 9999
            maxY = 0

            for k in h1_parts:

                body_part = h1[k]
                minX = int(body_part[0]*IM_W) if minX > int(body_part[0]*IM_W) else minX
                maxX = int(body_part[0]*IM_W) if maxX < int(body_part[0]*IM_W) else maxX
                minY = int(body_part[1]*IM_H) if minY > int(body_part[1]*IM_H) else minY
                maxY = int(body_part[1]*IM_H) if maxY < int(body_part[1]*IM_H) else maxY 

            tl = (minX, minY)
            br = (maxX, maxY)

            area = (maxY - minY) * (maxX - minX)
            rospy.loginfo('area h1 = %d' % area)
            # print("H1_PARTS : ",h1_parts)

            for h2_part in h2_parts:
                if(h2_part not in h1_parts):
                    continue

                # Compute distance
                h2_p = h2[h2_part]
                h1_p = h1[h2_part]

                dist = math.sqrt((h2_p[0] - h1_p[0])
                                 ** 2 + (h2_p[1] - h1_p[1]) ** 2)

                if(dist <= DIST_THR):
                    inter_size += 1

                # print("dist(%2d) = %1.4f" %(h2_part, dist))

            # (len(h2_parts) + len(h1_parts)) - inter_size
            union_size = 10 - inter_size

            # Jaccard
            jac_sim = 1.0 * inter_size / union_size
            jacc_list.append(jac_sim)
            # print("Jaccard : ",jac_sim)

        # print("Jaccard List : ",jacc_list)
        best_jacc = 0.0
        if(len(humans1)):
            best_jacc = max(jacc_list)
        if(len(humans1) and best_jacc > JAC_THR):
            max_index = jacc_list.index(best_jacc)  # max jaccard score
            # Merge the two readings
            for i in parts_names.keys():
                # if i in h1.body_parts.keys() and i in h2.body_parts.keys():
                if i in h2.keys():
                    # print("Merging : ", i)
                    humans1[max_index][i] = h2[i]
        else:
            # print("Adding H2")
            new_humans.append(h2)

    return humans1 + new_humans


def humans_to_msg(humans):
    persons = Persons()

    for human in humans:
        person = Person()

        for k in human.keys():
            body_part = human[k]

            body_part_msg = BodyPartElm()
            body_part_msg.part_id = k
            body_part_msg.x = body_part[0]
            body_part_msg.y = body_part[1]
            body_part_msg.confidence = body_part[2]
            person.body_part.append(body_part_msg)
        persons.persons.append(person)

    return persons


def msg_to_humans(msg):
    humans = []
    for person in msg.persons:
        human = {}
        for body_part in person.body_part:
            human[body_part.part_id] = [body_part.x, body_part.y,
                                        body_part.confidence, parts_names[body_part.part_id]]
        humans.append(human)
    return humans


def callback(p1_msg, p2_msg, p3_msg):

    #pub_next

    global assoc_index_thr
    global part_num_thr

    t1 = time.time()
    humans1 = msg_to_humans(p1_msg) + msg_to_humans(p3_msg)
    humans2 = msg_to_humans(p2_msg)
    #humans3 = msg_to_humans(p3_msg)
    rospy.loginfo('msg_to_humans time=%.5f' % (time.time() - t1))
    t = time.time()

    humans = associate_humans(humans2, humans1, p1_msg.image_w, p1_msg.image_h, part_num_thr, assoc_index_thr)
    #humans = associate_humans(humans, humans3, p1_msg.image_w, p1_msg.image_h, part_num_thr, assoc_index_thr)
    
    rospy.loginfo('associate_humans time=%.5f' % (time.time() - t))
    
    t = time.time()
    msg = humans_to_msg(humans)
    rospy.loginfo('humans_to_msg time=%.5f' % (time.time() - t))
    msg.header = p1_msg.header
    msg.image_w = p1_msg.image_w
    msg.image_h = p1_msg.image_h
    msg.crop_w = p2_msg.crop_w
    msg.crop_h = p2_msg.crop_h
    msg.crop_x = p2_msg.crop_x
    msg.crop_y = p2_msg.crop_y

    pub_pose.publish(msg)
    pub_next_frame.publish(std_msgs.msg.String("hello"))

    rospy.loginfo('MERGER time=%.5f' % (time.time() - t1))

def callback2(p2_msg):

    global assoc_index_thr
    global part_num_thr

    t1 = time.time()
    humans2 = msg_to_humans(p2_msg)
    rospy.loginfo('msg_to_humans time=%.5f' % (time.time() - t1))
    t = time.time()

    rospy.loginfo('associate_humans time=%.5f' % (time.time() - t))
    
    t = time.time()
    msg = humans_to_msg(humans2)
    rospy.loginfo('humans_to_msg time=%.5f' % (time.time() - t))
    msg.header = p2_msg.header
    msg.image_w = p2_msg.image_w
    msg.image_h = p2_msg.image_h
    msg.crop_w = p2_msg.crop_w
    msg.crop_h = p2_msg.crop_h
    msg.crop_x = p2_msg.crop_x
    msg.crop_y = p2_msg.crop_y

    pub_pose.publish(msg)

    rospy.loginfo('MERGER time=%.5f' % (time.time() - t1))


if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('TfPoseEstimatorROS-Merger', anonymous=True)

    # topics params
    poseA_topic = rospy.get_param('~poseA', '/poseA_estimator/poseA')
    poseB_topic = rospy.get_param('~poseB', '/poseB_estimator/poseB')
    poseC_topic = rospy.get_param('~poseC', '/poseC_estimator/poseC')

    # publishers
    pub_pose = rospy.Publisher('~pose', Persons, queue_size=1)

    pub_next_frame = rospy.Publisher('/next_frame', std_msgs.msg.String, queue_size=1)

    # association parameters
    global assoc_index_thr
    assoc_index_thr = rospy.get_param('~assoc_index_thr', 0.35)
    global part_num_thr
    part_num_thr = rospy.get_param('~part_num_thr', 10)

    # subscribers
    poseA_sub = message_filters.Subscriber(poseA_topic, Persons)
    poseB_sub = message_filters.Subscriber(poseB_topic, Persons)
    poseC_sub = message_filters.Subscriber(poseC_topic, Persons)

    #ts = message_filters.ApproximateTimeSynchronizer([poseA_sub, poseB_sub, poseC_sub], 20, 0.2)
    ts = message_filters.TimeSynchronizer([poseA_sub, poseB_sub, poseC_sub], 4000)
    #ts = message_filters.TimeSynchronizer([poseB_sub], 50)
    ts.registerCallback(callback)

    time.sleep(5)

    pub_next_frame.publish(std_msgs.msg.String("a"))
    pub_next_frame.publish(std_msgs.msg.String("b"))
    pub_next_frame.publish(std_msgs.msg.String("c"))

    #pub_next = rospy.Publisher('/next_frame', std_msgs.Empty, queue_size=1)

    # run
    rospy.spin()
