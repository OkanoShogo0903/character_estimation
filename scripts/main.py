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

from sklearn.ensemble import RandomForestClassifier as classifier

CHARACTER_LABEL = [
        "Leone      Abbacchio",
        "Narancia   Ghirga",
        "Bruno      Bucciarati",
        "Giorno     Giovanna",
        "Guido      Mista",
        "Pannacotta Fugo",
        ]

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
        self.IS_TEST = rospy.get_param(base + '/is_test')
        print(self.IS_TEST)
        self.DATASET_URL = rospy.get_param(base + '/dataset_url')
        self.DATASET_FILENAME = rospy.get_param(base + '/dataset_filename')

        self.PACKAGE_PATH = rospkg.RosPack().get_path('character_estimation')
        # ROS Subscriber ----->>>
        self.joint_angle = rospy.Subscriber('/joint_angle', Joints, self.jointCB)
        # ROS Publisher ------>>>
        self.pose_label = rospy.Publisher('/pose_label', Int32MultiArray, queue_size=1)
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
            labels = self.detect(msg)
            self.outputter(labels)
        else:
            rospy.loginfo("MODE ERROR")
        

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


    def detect(self, msg):
        # Learn in first call --->
        if self.isFirstCall == False:
            print("Learnning --->>>")
            self.isFirstCall = True

            X = self.df.loc[:, 'R-mid-0':'L-bot-2']
            y = self.df.loc[:, 'Label']
            print(X.shape)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=0)

            from sklearn.ensemble import RandomForestClassifier
            print("X shape", X_train.shape)
            print(type(X_train))
            self.model = RandomForestClassifier(random_state=777)

            self.model.fit(X_train, y_train)
            #predict_y_train = self.model.predict(X_train)
            #predict_y_test  = self.model.predict(X_test)
   	    print('score_train: {}'.format(self.model.score(X_train, y_train)))
            print('score_test : {}'.format(self.model.score(X_test, y_test)))
        #self.IS_TEST

        # Save model --->
        # Estimate --->
        labels = []
        for person in msg.persons:
            input_data = np.reshape(np.array(person.data), (1,16))
            predict = self.model.predict(input_data)
            print("Detect ---> ", CHARACTER_LABEL[int(predict)])
            labels.append(predict)

        return labels


    def outputter(self, labels):
        pass
        
        
# [Main] ----------------------------------------->>>
#if __name__ == '__main__':
rospy.init_node('character_estimation')
rospy.sleep(3)

node = CharacterEstimatate()

while not rospy.is_shutdown():
    rospy.sleep(0.1)
            
