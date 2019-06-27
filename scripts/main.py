#!/usr/bin/python
# -*- coding: utf-8 -*-

# [My Lib]------------------------------->
import pil_image
import colum as col
import character as ch 

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

# [Define]------------------------------->
MODE = {
        "CREATE_DATASET": 0, 
        "TRAIN": 1, 
        "PREDICT": 2,
        }

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
        # ROS Basic setting ----->>>
        self.bridge = CvBridge()
        base = rospy.get_namespace() + rospy.get_name()
        self.select_mode    = rospy.get_param(base + '/mode') # 0:CreateDataset ,1:Train, 2:Predict
        self.IS_FLIP_ENABLE = rospy.get_param(base + '/flip_enable')
        self.DATASET_URL    = rospy.get_param(base + '/dataset_url')
        self.DATASET_FILENAME = rospy.get_param(base + '/dataset_filename')
        self.MODEL_URL      = rospy.get_param(base + '/model_url')
        self.PICTURE_URL    = rospy.get_param(base + '/picture_url')

        self.PACKAGE_PATH = rospkg.RosPack().get_path('character_estimation')

        print("select_mode ", self.select_mode)

        # Other setting ------>>>
        self.img = None
        self.person_x, self.person_y = [], []
        self.HEAD_ID = 0
        self.MODEL_NAME = "model.pkl"
        self.isRecording = False

        # Dataset load ----->>>
        try:
            self.df = pd.read_csv( self.DATASET_URL + '/' + self.DATASET_FILENAME, names=col.COLUMNS)
            rospy.loginfo("Load csv file")
        except IOError:
            self.df = pd.DataFrame( columns=col.COLUMNS)
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
        
        # For create create
        self.record_count = 0

    def shutdown(self):
        ''' 
        @dis This function always call in shutdown
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
        @dis This function receive ros-openpose msg.
             We want to only use head coordinate.
        ''' 
        # Delete old data.
        del self.person_x[:]
        del self.person_y[:]
        # Set new data(head coordinate).
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
            self.predict(msg)
            self.publishCharacterImage()
        else:
            rospy.loginfo("MODE ERROR")
        

    def createDataset(self, msg):
        ''' 
        @dis In first call, receive class label in keyboard.
        ''' 
        # Get data label --->
        if self.isRecording == False:
            print("\007")
            print("Please enter the label --->>>")
            print("Please enter the label --->>>")
            print("Please enter the label --->>>")
            print("Please enter the label --->>>")
            print("Please enter the label --->>>")
            self.label = input()
            print("Recording start after some second latter")
            for _ in range(0,5): # Wait some second
                print(".")
                rospy.sleep(1.0)
            self.isRecording = True
        else:
            # Append to dataset --->
            for p in msg.persons:
                data_raw = np.hstack((p.angle.data, p.length.data, self.label))
                tmp_se = pd.Series( data_raw, index=self.df.columns )
                self.df = self.df.append( tmp_se, ignore_index=True )

            # Count up
            self.record_count += 1
            if self.record_count % 100 == 0:
                self.isRecording = False


    def predict(self, msg):
        # Load model
        os.chdir(self.MODEL_URL)
        model = load(self.MODEL_NAME)

        labels = []
        for p in msg.persons:
            # Reshape input data --->
            data = p.angle.data + p.length.data
            data = np.reshape(np.array(data), (1,16+17))
            data = pd.DataFrame(data, columns=col.COLUMNS[:-1])
            data = data.drop(col.DROP_LIST, axis=1)

            # Predict --->
            predict = model.predict(data)
            labels.append(predict)
            print("Detected ---> ", ch.CHARACTER_NAME[int(predict)])

            # Pose label publish --->
            label = Int32()
            label.data = predict
            self.pose_label_pub.publish(label)

        self.labels = labels


    def dataVisualize(self, df):
        df.plot.scatter(x='L-top-1', y='Label', vmin=0, vmax=180)
        plt.show()


    def dataAugmentation(self, df):
        '''
        Replace the columns
        '''
        if self.IS_FLIP_ENABLE == True:
            copy = df.copy()
            copy.columns = col.FLIP_COLUMNS
            return pd.concat([df, copy], sort=True)
        else:
            return df


    def publishCharacterImage(self):
        from PIL import Image, ImageDraw, ImageFilter

        # Publish Image with label
        img = self.img
        for x, y, label in zip(self.person_x, self.person_y, self.labels):
            # Draw character image --->
            os.chdir(self.PICTURE_URL)
            char = ch.CHARACTER_PICTURE[int(label)]

            # Load and resize
            src = pil_image.cv2pil(img)
            forward = Image.open( char["filename"])
            h, w = forward.size
            forward = forward.resize((int(h*char["size"]), int(w*char["size"])))

            # For convert
            src = src.convert('RGBA')

            # Paste
            c = Image.new('RGBA', src.size, (255, 255, 255, 0))
            #src.paste(forward, ( char["x"], char["y"]))
            #c.paste(forward, ( char["x"] + x, char["y"] + y), forward)
            c.paste(forward, ( char["x"] + x - int(char["size"]*h/2), char["y"] + y - int(char["size"]*w/2)), forward)

            result = Image.alpha_composite(src, c)
            result.save('tmp.jpg', quality=95)
            img = cv.imread('tmp.jpg')

            # Write label name to img
            cv.putText(img, ch.CHARACTER_NAME[int(label)], (x-50,y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,150), 2, cv.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
        self.image_pub.publish(msg)
        

    def train(self):
        print("Learnning --->>>")
        os.chdir(self.MODEL_URL)

        # 1. Load dataset --->
        data = self.dataAugmentation(self.df)
        #self.dataVisualize(data)
        data = data.drop(col.DROP_LIST, axis=1)
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

        dump(self.model, self.MODEL_NAME, compress=True) # Save

        # 3. evaluating the performance of the self.model --->
        print('score_train : {}'.format(self.model.score(X_train, y_train)))
        print('score_test  : {}'.format(self.model.score(X_test, y_test)))

        for x, y in zip([X_train, X_test], [y_train, y_test]):
            print("-"*50)
            y_pred = self.model.predict(x)
            print(classification_report(y, y_pred, target_names=ch.CHARACTER_NAME))

        # 4. printing parameters of the model --->
        print(sorted(self.model.get_params(True).items()))
         
        # 5. printing importances of the model --->
        #print(self.model.feature_importances_)
        features = X_train.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(6,6))
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), features[indices])
        os.chdir(self.MODEL_URL)
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
