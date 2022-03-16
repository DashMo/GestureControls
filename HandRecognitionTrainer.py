import mediapipe as mp
import numpy
import cv2
import os
import numpy as np
import pickle
import HandTrackingModule as hmd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class HandRecognitionTrainer():
    def __init__(self,model):
        self.detector = hmd.handDetector(detectionCon=.5)
        self.model = model
        self.rawCoordsNorm = []
        self.handShape = []
        self.numFailedToFind
        self.failedToFind = []
    
    def addDataHelper(self,img, shape, file):
        self.detector.findHands(img,draw=False)
        if self.detector.results.multi_hand_landmarks:
            for handNum in range(2):
                lms, bbox = self.detector.findPositionAdvanced(img,[],drawBoundingBox=False,drawFeatures=False,handNo=handNum)
                if lms:
                    thumb_id,thumb_cx,thumb_cy,thumb_x,thumb_y,thumb_z = lms[0]
                    featRawCoordsNorm = []
                    xmin, ymin = min(lms,key=lambda x: x[3])[3], min(lms,key=lambda x: x[4])[4]
                    xmax, ymax = max(lms,key=lambda x: x[3])[3], max(lms,key=lambda x: x[4])[4]
                    for i,lm in enumerate(lms):
                        id,cx,cy,x,y,z = lm
                        newX = np.interp(x, (xmin, xmax), (0, 1))
                        newY = np.interp(y, (ymin, ymax), (0, 1))
                        featRawCoordsNorm.extend((newX,newY,z))
                self.rawCoordsNorm.append(featRawCoordsNorm)
                self.handShape.append(shape)
        else:
            self.numFailedToFind +=1
            self.failedToFind.append(file)
    
    def addData(self,image_file_list, prefix=""):
        for idx,file in enumerate(image_file_list):
            shape = int(file.split('_')[0])
            
            img = cv2.imread(prefix + file)
            self.addDataHelper(img, shape, file)
            
            img = cv2.flip(img,1)
            self.addDataHelper(img, shape, file)
            
    def fit(self, test_size = .8):
        rawCoordsArray = np.array(self.rawCoordsNorm)
        handShapeArray = np.array(self.handShape)
        X_train, X_test, y_train, y_test = train_test_split(rawCoordsArray, handShapeArray, stratify=handShapeArray,random_state=1, test_size=test_size)
        self.model.fit(X_train, y_train)
        
        print(f"The model has an accuracy of {self.model.score(X_test, y_test)*100}% on the test data") 
        
    def saveModel(self,filepath):
       with open(filepath,'wb') as f:
            pickle.dump(self.model,f) 