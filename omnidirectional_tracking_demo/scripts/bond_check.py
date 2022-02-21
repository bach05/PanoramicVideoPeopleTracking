#!/usr/bin/env python

import rospy
from bondpy import bondpy


if __name__ == '__main__':
    rospy.loginfo('initialization+')
    rospy.init_node('BondChecker', anonymous=True, log_level=rospy.INFO)

    id = '10'
    # Sends id to B using an action or a service
    bond = bondpy.Bond("example_bond_topic", id)
    bond.start()
    if not bond.wait_until_formed(rospy.Duration(10.0)):
        raise Exception('Bond could not be formed')
    # ... do things with B ... 
    bond.wait_until_broken()
    print("B has broken the bond")