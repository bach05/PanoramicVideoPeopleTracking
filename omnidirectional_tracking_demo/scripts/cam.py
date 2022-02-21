#!/usr/bin/env python

import rospy
import time
from sensor_msgs.msg import Image
import os
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError


def getPerspective(img, FOV, THETA, PHI, height, width, RADIUS=128):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    equ_h = img.shape[0]
    equ_w = img.shape[1]
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    wFOV = FOV
    hFOV = float(height) / width * wFOV

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (180 - wFOV) / 2.0
    w_len = 2 * RADIUS * \
        np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * \
        np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
    h_interval = h_len / (height - 1)

    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y)
                     * h_interval, [width, 1]).T
    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.zeros([height, width, 3], np.float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

    xyz = xyz.reshape([height * width, 3]).T
    xyz = np.dot(R1, xyz)
    xyz = np.dot(R2, xyz).T
    lat = np.arcsin(xyz[:, 2] / RADIUS)
    lon = np.zeros([height * width], np.float)
    theta = np.arctan(xyz[:, 1] / xyz[:, 0])
    idx1 = xyz[:, 0] > 0
    idx2 = xyz[:, 1] > 0

    idx3 = ((1 - idx1) * idx2).astype(np.bool)
    idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

    lon[idx1] = theta[idx1]
    lon[idx3] = theta[idx3] + np.pi
    lon[idx4] = theta[idx4] - np.pi

    lon = lon.reshape([height, width]) / np.pi * 180
    lat = -lat.reshape([height, width]) / np.pi * 180
    lon = lon / 180 * equ_cx + equ_cx
    lat = lat / 90 * equ_cy + equ_cy

    persp = cv2.remap(img, lon.astype(np.float32), lat.astype(
        np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return persp


rospy.init_node('theta_rect', anonymous=True)

VideoRaw = rospy.Publisher('raw', Image, queue_size=10)

theta = cv2.VideoCapture(5)

ret, frame = theta.read()

image_size = frame.shape[:-1]

while(True):

    # Capture the video frame
    # by frame
    ret, frame = theta.read()

    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension

    # Specify parameters(FOV, theta, phi, height, width)
    frame = getPerspective(frame, 60, 180, -40, 720, 720)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8")
    VideoRaw.publish(msg_frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)

# After the loop release the cap object
theta.release()
# Destroy all the windows
cv2.destroyAllWindows()
