#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from monocular_people_tracking.msg import *

def callback(data):
    text = 'frame%d ' % (data.header.seq)
    for track in data.tracks:
        text += '%f %f %f %f +' % (track.pos.x, track.pos.y, track.pos.z, track.height)
    rospy.loginfo(text)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('/monocular_people_tracking/tracks', TrackArray, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
