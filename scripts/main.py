#!/usr/bin/python
# -*- coding: utf-8 -*-

# [Import]------------------------------->
import os
import sys
import time
import json
import math
import types
import numpy as np 
import pandas as pd

import rospy
import rospkg
from std_msgs.msg import Int32MultiArray
from openpose_ros_msgs.msg import Persons
from ros_openpose_joint_converter.msg import Joints

# [ClassDefine]-------------------------->
class CharacterEstimatate():
    ''' 
    This class receive human joint angle from openpose.
    It detect character label.
    '''
    def __init__(self):
        # ROS Params ----->>>
        base = rospy.get_namespace() + rospy.get_name()
        self.MODE = rospy.get_param(base + '/mode', "create_dataset")
        self.DATASET_URL = rospy.get_param(base + '/dataset_url')
        self.DATASET_FILENAME = rospy.get_param(base + '/dataset_filename')

        self.PACKAGE_PATH = rospkg.RosPack().get_path('character_estimation')
        # ROS Subscriber ----->>>
        self.joint_angle = rospy.Subscriber('/joint_angle', Joints, self.jointCB)
        # ROS Publisher ------>>>
        #self.joint_angle = rospy.Publisher('/joint_angle', Int32MultiArray, queue_size=1)
        # Set rospy to execute a shutdown function when exiting --->
        rospy.on_shutdown(self.shutdown)

        print("MODE ", self.MODE)

        # Other setting ------>>>
        self.isFirstCall = False
        columns=[
            'R-mid-0','R-mid-1','R-mid-2',
            'L-mid-0','L-mid-1','L-mid-2',
            'R-top-0','R-top-1',
            'L-top-0','L-top-1',
            'R-bot-0','R-bot-1','R-bot-2',
            'L-bot-0','L-bot-1','L-bot-2',
            'Label']

        os.chdir(self.DATASET_URL)
        try:
            self.df = pd.read_csv( self.DATASET_FILENAME, names=columns)
            rospy.loginfo("Load csv file")
        except IOError:
            self.df = pd.DataFrame( columns=columns)
            rospy.loginfo("Could not find csv file")
            rospy.loginfo("Create new csv file...")


    def shutdown(self):
        ''' 
        This function always call in shutdown
        ''' 
        rospy.loginfo("Stopping the system...")
        # 
        if self.MODE == "create_dataset":
            rospy.loginfo("Save csv data")
            os.chdir(self.DATASET_URL)
            self.df.to_csv( self.DATASET_FILENAME, encoding="utf-8", header=False, index=False)
            #self.df.to_csv( "hoge.csv", encoding="utf-8")
            rospy.loginfo("Successful")


    def jointCB(self, msg):
        ''' 
        @param msg : it is joint angle from ros_openpose_joint_converter.
        ''' 
        if self.MODE == "create_dataset":
            self.createDataset(msg)
        elif self.MODE == "estimate":
            pass
        else:
            print("error")
        

    def createDataset(self, msg):
        ''' 
        @dis 
        ''' 
        # Get data label --->
        if self.isFirstCall == False:
            print("Please enter the label --->>>")
            self.label = input()
            self.isFirstCall = True
        # Save dataset --->
        for person in msg.persons:
            data_raw = np.hstack((person.data, self.label))
            tmp_se = pd.Series( data_raw, index=self.df.columns )
            self.df = self.df.append( tmp_se, ignore_index=True )

            # PACKAGE_PATH/dataset
        print(self.df)


    def outputter(self):
        pass
        

# [Main] ----------------------------------------->>>
#if __name__ == '__main__':
rospy.init_node('character_estimation')

time.sleep(3.0)
node = CharacterEstimatate()

while not rospy.is_shutdown():
    rospy.sleep(0.1)
            
