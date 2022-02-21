#!/usr/bin/env python

import rospy
import math
from std_msgs.msg import String
from monocular_people_tracking.msg import *
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import CompressedImage

def callback_tracks(msg):

    # new_dist = old_dist * (camera_height - ankle_offset) / camera_height
    k = (camera_height - ankle_offset) / camera_height
    for track in msg.tracks:

        x = track.pos.x
        y = track.pos.y
        z = track.height
        sx = x/abs(x)
        sy = y/abs(y)
        m = abs(x)/abs(y)
        new_y = math.sqrt((x**2+y**2)*k**2 / (m**2+1))
        new_x = m*new_y
        
        track.pos.x = new_x * sx
        track.pos.y = new_y * sy
        new_z = camera_height + (((z-camera_height)*(math.sqrt(track.pos.x**2+track.pos.y**2))) / math.sqrt(x**2+y**2))
        track.height = new_z

    pub_tracks.publish(msg)

def callback_markers(msg):
    k = (camera_height - ankle_offset) / camera_height
    for point in msg.markers[0].points:

        x = point.x
        y = point.y
        sx = x/abs(x)
        sy = y/abs(y)
        m = abs(x)/abs(y)
        new_y = math.sqrt((x**2+y**2)*k**2 / (m**2+1))
        new_x = m*new_y
        point.x = new_x * sx
        point.y = new_y * sy

    for point in msg.markers[1].points:

        x = point.x
        y = point.y
        sx = x/abs(x)
        sy = y/abs(y)
        m = abs(x)/abs(y)
        new_y = math.sqrt((x**2+y**2)*k**2 / (m**2+1))
        new_x = m*new_y
        point.x = new_x * sx
        point.y = new_y * sy

    pub_markers.publish(msg)


if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('AnkleCorrection', anonymous=True, log_level=rospy.INFO)

    camera_height = rospy.get_param('~camera_height', '1.072')
    ankle_offset = rospy.get_param('~ankle_offset', '0.10')

    # Published topics
    tracks_pub = '/monocular_people_tracking/tracks_corrected'
    markers_pub = '/monocular_people_tracking/markers_corrected'
    
    rospy.Subscriber('/monocular_people_tracking/tracks', TrackArray, callback_tracks, queue_size=10)
    rospy.Subscriber('/monocular_people_tracking/markers', MarkerArray, callback_markers, queue_size=10)

    pub_tracks = rospy.Publisher(tracks_pub, TrackArray, queue_size=10)
    pub_markers = rospy.Publisher(markers_pub, MarkerArray, queue_size=10)

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
