#!/usr/bin/python
# -*- coding: utf-8 -*-

# [Import]------------------------------->
import os
import cv2 as cv
import sys
import time
import json
import math
import types
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt

import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Int32MultiArray, Int32
from sensor_msgs.msg import Image, CameraInfo

from openpose_ros_msgs.msg import Persons
from ros_openpose_joint_converter.msg import Joints

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

CHARACTER_NAME = [
        "Leone Abbacchio",
        "Narancia Ghirga",
        "Bruno Bucciarati",
        "Giorno Giovanna",
        "Guido Mista",
        "Pannacotta Fugo",
        ]

clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50),
        KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
        GaussianNB(),
        XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1,
              gamma=0, subsample=0.8, colsample_bytree=0.5, objective= 'binary:logistic',
              scale_pos_weight=1, seed=0
             )
       ]

# [ClassDefine]-------------------------->
class CharacterEstimatate():
    ''' 
    This class receive human joint angle from openpose.
    It detect character label.
    '''
    def __init__(self):
        # ROS Basic ----->>>
        self.bridge = CvBridge()
        base = rospy.get_namespace() + rospy.get_name()
        self.MODE = rospy.get_param(base + '/mode', "create_dataset")
        self.IS_TEST = rospy.get_param(base + '/is_test')
        self.DATASET_URL = rospy.get_param(base + '/dataset_url')
        self.DATASET_FILENAME = rospy.get_param(base + '/dataset_filename')

        self.PACKAGE_PATH = rospkg.RosPack().get_path('character_estimation')
        # ROS Subscriber ----->>>
        self.joint_sub = rospy.Subscriber('/joint_angle', Joints, self.jointCB)
        self.pose_sub  = rospy.Subscriber('/openpose/pose', Persons, self.poseCB)
        self.image_sub = rospy.Subscriber('/videofile/image_raw', Image, self.imageCB)
        # ROS Publisher ------>>>
        self.pose_label_pub = rospy.Publisher('/pose_label', Int32, queue_size=1)
        self.image_pub = rospy.Publisher('/character_img', Image, queue_size=1)
        # Set rospy to execute a shutdown function when exiting --->
        rospy.on_shutdown(self.shutdown)

        print("MODE ", self.MODE)

        # Other setting ------>>>
        self.img = None
        self.person_x, self.person_y = [], []
        self.HEAD_ID = 0
        self.isFirstCall = False
        self.labels = []
        self.COLUMNS=[
            'R-mid-0','R-mid-1','R-mid-2',
            'L-mid-0','L-mid-1','L-mid-2',
            'R-top-0','R-top-1',
            'L-top-0','L-top-1',
            'R-bot-0','R-bot-1','R-bot-2',
            'L-bot-0','L-bot-1','L-bot-2',
            'Label']

        self.FLIP_COLUMNS=[
            'L-mid-0','L-mid-1','L-mid-2',
            'R-mid-0','R-mid-1','R-mid-2',
            'L-top-0','L-top-1',
            'R-top-0','R-top-1',
            'L-bot-0','L-bot-1','L-bot-2',
            'R-bot-0','R-bot-1','R-bot-2',
            'Label']

        os.chdir(self.DATASET_URL)
        try:
            self.df = pd.read_csv( self.DATASET_FILENAME, names=self.COLUMNS)
            rospy.loginfo("Load csv file")
        except IOError:
            self.df = pd.DataFrame( columns=self.COLUMNS)
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


    def poseCB(self, msg):
        ''' 
        Update pose data.
        ''' 
        #self.person_x.clear()
        #self.person_y.clear()
        del self.person_x[:]
        del self.person_y[:]
        for person in msg.persons:
            self.person_x.append(person.body_part[self.HEAD_ID].x)
            self.person_y.append(person.body_part[self.HEAD_ID].y)


    def imageCB(self, msg):
        ''' 
        Update image data.
        ''' 
        # Image convert from ROS IMAGE format to OPENCV IMAGE format ----->
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            #self.img = self.bridge.imgmsg_to_cv2(msg, msg.encoding)

        except CvBridgeError as e:
            print(e)
            return


    def jointCB(self, msg):
        ''' 
        @param msg : it is joint angle from ros_openpose_joint_converter.
        ''' 
        if self.MODE == "create_dataset":
            self.createDataset(msg)
        elif self.MODE == "estimate":
            self.labels = self.detect(msg)
            #if self.img != None:
            self.outputter()
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


    def detect(self, msg):
        # Train in first call --->
        if self.isFirstCall == False:
            print("Learnning --->>>")
            self.isFirstCall = True

            # Dataset --->
            data = self.df
            #data = self.dataAugmentation(data)
            print("Dataset shape :", data)
            data = self.df.drop(['R-top-0', 'R-top-1', 'L-top-0', 'L-top-1'], axis=1)
            print("Dataset shape :", data)
            y = data.loc[:, 'Label']
            X = data.drop('Label', axis=1)
            #X = argmentted_df.loc[:, 'R-mid-0':'L-bot-2']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=0)

            # Train --->
            self.model = clfs[0]
            self.model.fit(X_train, y_train)

            # Evaluation --->
   	    print('score_train : {}'.format(self.model.score(X_train, y_train)))
            print('score_test  : {}'.format(self.model.score(X_test, y_test)))
            #print('Accracy     : {}'.format(accuracy_score(X_test, y_test)))
            #print('Precision   : {}'.format(precision_score(X_test, y_test)))
            #print('Recall      : {}'.format(recall_score(X_test, y_test)))
            #print('F-score     : {}'.format(f1_score(X_test, y_test)))

        # Save model --->
        # Estimate --->
        labels = []
        for person in msg.persons:
            data = person.data
            data = np.reshape(np.array(data), (1,16))
            data = pd.DataFrame(data, columns=self.COLUMNS[:-1])
            data = data.drop(['R-top-0', 'R-top-1', 'L-top-0', 'L-top-1'], axis=1)

            predict = self.model.predict(data)
            print("Detect ---> ", CHARACTER_NAME[int(predict)])
            labels.append(predict)

            # Pose label publish --->
            label = Int32()
            label.data = predict
            self.pose_label_pub.publish(label)

        return labels


    def dataAugmentation(self, df):
        copy = df.copy()
        copy.columns = self.FLIP_COLUMNS
        return pd.concat([df, copy])


    def outputter(self):
        img = self.img
        for x, y, label in zip(self.person_x, self.person_y, self.labels):
            cv.putText(img, CHARACTER_NAME[int(label)], (x-50,y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,150), 2, cv.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        self.image_pub.publish(msg)
        
        
# [Main] ----------------------------------------->>>
#if __name__ == '__main__':
rospy.init_node('character_estimation')
rospy.sleep(2)

node = CharacterEstimatate()
rospy.sleep(2)

while not rospy.is_shutdown():
    rospy.sleep(0.1)
            
