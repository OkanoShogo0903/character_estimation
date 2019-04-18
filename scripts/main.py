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
from ros_openpose_joint_converter.msg import Group, Person

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.joblib import dump, load
from sklearn.metrics import confusion_matrix, classification_report

MODE = {
        "CREATE_DATASET": 0, 
        "TRAIN": 1, 
        "PREDICT": 2,
        }

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
        self.select_mode = rospy.get_param(base + '/mode') # 0:CreateDataset ,1:Train, 2:Predict
        self.DATASET_URL = rospy.get_param(base + '/dataset_url')
        self.DATASET_FILENAME = rospy.get_param(base + '/dataset_filename')

        self.PACKAGE_PATH = rospkg.RosPack().get_path('character_estimation')

        print("select_mode ", self.select_mode)

        # Other setting ------>>>
        self.img = None
        self.person_x, self.person_y = [], []
        self.HEAD_ID = 0
        self.isRecording = False
        #self.labels = []
        self.COLUMNS=[
            'angle-R-mid-0', 'angle-R-mid-1', 'angle-R-mid-2',
            'angle-L-mid-0', 'angle-L-mid-1', 'angle-L-mid-2',
            'angle-R-top-0', 'angle-R-top-1',  
            'angle-L-top-0', 'angle-L-top-1',  
            'angle-R-bot-0', 'angle-R-bot-1', 'angle-R-bot-2',
            'angle-L-bot-0', 'angle-L-bot-1', 'angle-L-bot-2',
            'length-C-top-0',
            'length-R-mid-0', 'length-R-mid-1', 'length-R-mid-2',
            'length-L-mid-0', 'length-L-mid-1', 'length-L-mid-2',
            'length-R-bot-0', 'length-R-bot-1', 'length-R-bot-2',
            'length-L-bot-0', 'length-L-bot-1', 'length-L-bot-2',
            'length-R-top-0', 'length-R-top-1', 
            'length-L-top-0', 'length-L-top-1',  
            'Label']

        self.FLIP_COLUMNS=[
            'angle-L-mid-0', 'angle-L-mid-1', 'angle-L-mid-2',
            'angle-R-mid-0', 'angle-R-mid-1', 'angle-R-mid-2',
            'angle-L-top-0', 'angle-L-top-1',  
            'angle-R-top-0', 'angle-R-top-1',  
            'angle-L-bot-0', 'angle-L-bot-1', 'angle-L-bot-2',
            'angle-R-bot-0', 'angle-R-bot-1', 'angle-R-bot-2',
            'length-C-top-0',
            'length-R-mid-0', 'length-R-mid-1', 'length-R-mid-2',
            'length-L-mid-0', 'length-L-mid-1', 'length-L-mid-2',
            'length-R-bot-0', 'length-R-bot-1', 'length-R-bot-2',
            'length-L-bot-0', 'length-L-bot-1', 'length-L-bot-2',
            'length-R-top-0', 'length-R-top-1', 
            'length-L-top-0', 'length-L-top-1',  
            'Label']

        self.DROP_LIST=[
            'angle-L-top-0', 'angle-L-top-1',  
            'angle-R-top-0', 'angle-R-top-1',  
            'length-R-top-0', 'length-R-top-1', 
            'length-L-top-0', 'length-L-top-1',  
            ]
        os.chdir(self.DATASET_URL)
        try:
            self.df = pd.read_csv( self.DATASET_FILENAME, names=self.COLUMNS)
            rospy.loginfo("Load csv file")
        except IOError:
            self.df = pd.DataFrame( columns=self.COLUMNS)
            rospy.loginfo("Could not find csv file")
            rospy.loginfo("Create new csv file...")

        # ROS Subscriber ----->>>
        self.joint_sub = rospy.Subscriber('/joint_converter/group', Group, self.jointCB)
        self.pose_sub  = rospy.Subscriber('/openpose/pose', Persons, self.poseCB)
        #self.image_sub = rospy.Subscriber('/videofile/image_raw', Image, self.imageCB)
        self.image_sub = rospy.Subscriber('/openpose/image_raw', Image, self.imageCB)
        # ROS Publisher ------>>>
        self.pose_label_pub = rospy.Publisher('/pose_label', Int32, queue_size=1)
        self.image_pub = rospy.Publisher('/character_img', Image, queue_size=1)
        # Set rospy to execute a shutdown function when exiting --->
        rospy.on_shutdown(self.shutdown)
        
        # hoge
        self.record_count = 0

    def shutdown(self):
        ''' 
        This function always call in shutdown
        ''' 
        rospy.loginfo("Stopping the system...")
        # 
        if self.select_mode == MODE["CREATE_DATASET"]:
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
        except CvBridgeError as e:
            print(e)
            return


    def jointCB(self, msg):
        ''' 
        @param msg : it is joint angle from ros_openpose_joint_converter.
        ''' 
        if self.select_mode == MODE["CREATE_DATASET"]:
            self.createDataset(msg)
        elif self.select_mode == MODE["TRAIN"]:
            pass
        elif self.select_mode == MODE["PREDICT"]:
            self.labels = self.predict(msg)
            self.outputter()
        else:
            rospy.loginfo("MODE ERROR")
        

    def createDataset(self, msg):
        ''' 
        @dis In first call, receive class label in keyboard.
        ''' 
        # Get data label --->
        if self.isRecording == False:
            print("Please enter the label --->>>")
            self.label = input()
            print("Recording start after some second latter")
            for _ in range(0,5): # Wait some second
                print(".")
                rospy.sleep(1.0)
            self.isRecording = True
        else:
            # Save dataset --->
            for p in msg.persons:
                data_raw = np.hstack((p.angle.data, p.length.data, self.label))
                tmp_se = pd.Series( data_raw, index=self.df.columns )
                self.df = self.df.append( tmp_se, ignore_index=True )

            # Count up
            self.record_count += 1
            if self.record_count % 100 == 0:
                self.isRecording = False


    def predict(self, msg):
        labels = []
        model = load("model.pkl")
        for p in msg.persons:
            # Reshape input data --->
            data = p.angle.data + p.length.data
            data = np.reshape(np.array(data), (1,16+17))
            data = pd.DataFrame(data, columns=self.COLUMNS[:-1])
            data = data.drop(self.DROP_LIST, axis=1)

            # Predict --->
            predict = model.predict(data)
            labels.append(predict)
            print("Detected ---> ", CHARACTER_NAME[int(predict)])

            # Pose label publish --->
            label = Int32()
            label.data = predict
            self.pose_label_pub.publish(label)

        return labels


    def dataVisualize(self, df):
        df.plot.scatter(x='L-top-1', y='Label', vmin=0, vmax=180)
        plt.show()
            #'L-top-0','',
            #'R-bot-0','R-bot-1','R-bot-2',
            #'L-bot-0','L-bot-1','L-bot-2',


    def dataAugmentation(self, df):
        '''
        Replace the columns
        '''
        copy = df.copy()
        copy.columns = self.FLIP_COLUMNS
        return pd.concat([df, copy], sort=True)


    def outputter(self):
        # Publish Image with label
        img = self.img
        for x, y, label in zip(self.person_x, self.person_y, self.labels):
            cv.putText(img, CHARACTER_NAME[int(label)], (x-50,y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,150), 2, cv.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        self.image_pub.publish(msg)
        
    def train(self):
        print("Learnning --->>>")

        # 1. Load dataset --->
        data = self.df
        #data = self.dataAugmentation(data)
        #self.dataVisualize(data)
        data = data.drop(self.DROP_LIST, axis=1)
        print("Dataset (Augmentation):", data)
        y = data.loc[:, 'Label']
        X = data.drop('Label', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
        y_train = np.ravel(y_train)
        y_test  = np.ravel(y_test)

        # 2. Train --->
        #self.model = clfs[0]
        diparameter = {
                "n_estimators":[i for i in range(10,50,10)], 
                "criterion":["gini","entropy"],
                "max_depth":[i for i in range(1,6,1)],
                "random_state":[123],
                "class_weight":["balanced"],
                }
        licv = GridSearchCV(RandomForestClassifier(), param_grid=diparameter, cv=5, n_jobs=5, verbose=2)
        licv.fit(X_train, y_train)
        self.model = licv.best_estimator_
        dump(self.model, "model.pkl", compress=True) # Save

        # 3. evaluating the performance of the self.model
        print('score_train : {}'.format(self.model.score(X_train, y_train)))
        print('score_test  : {}'.format(self.model.score(X_test, y_test)))

        for x, y in zip([X_train, X_test], [y_train, y_test]):
            print("-"*50)
            y_pred = self.model.predict(x)
            print(classification_report(y, y_pred, target_names=CHARACTER_NAME))

        # 4. printing parameters of the model
        print(sorted(self.model.get_params(True).items()))
         
        # 5. printing importances of the model
        #print(self.model.feature_importances_)
        features = X_train.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(6,6))
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.savefig('feature_importances.png')
        plt.show()

        
# [Main] ----------------------------------------->>>
#if __name__ == '__main__':
rospy.init_node('character_estimation')
rospy.sleep(2)

node = CharacterEstimatate()
if node.select_mode == MODE["TRAIN"]:
    node.train()
else:
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
