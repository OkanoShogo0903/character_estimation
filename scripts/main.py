#!/usr/bin/python
# -*- coding: utf-8 -*-

# [Import]------------------------------->
import sys
import json
import math
import time
import types
import threading
import numpy as np 
import math
from matplotlib import pyplot

import rospy
#from std_msgs.msg import String, Bool, UInt32MultiArray
from openpose_ros_msgs.msg import Persons
# [ImportScripts]------------------------------->


# [ClassDefine]-------------------------->
class CharacterEstimatate():
    ''' 
    This class receive human joint angle from openpose.
    It detect character label.
    '''
    def __init__(self):
        self.PACKAGE_PATH = rospkg.RosPack().get_path('character_estimation')
        # ROS Subscriber ----->>>
        self.arm_move_sub = rospy.Subscriber('/openpose/pose', Persons, self.detecterCB)

    def detecterCB(self, msg):
        ''' 
        @param msg : it is joint angle from openpose.
                     data format depend on ros-openpose.
        ''' 
        print(msg)

    def createDataset(self):
        ''' 
        @dis 
        ''' 
        # Change data format --->
        # Get data label --->
        # Save --->
        # PACKAGE_PATH/dataset

    def outputter(self):
        pass
        

# [Main] ----------------------------------------->>>
#if __name__ == '__main__':
rospy.init_node('display_disposal')

time.sleep(3.0)
node = CharacterEstimatate()

while not rospy.is_shutdown():
    rospy.sleep(0.1)
            
