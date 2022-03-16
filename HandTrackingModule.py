import cv2
import mediapipe as mp
import time
import numpy as np
import pickle
class handDetector():
    def __init__(self,mode=False, maxHands=2,complexity = 1, detectionCon=.5, trackCon=.5):
        """Utilizes mediapipe to process images to identify hands and hand features.

        Args:
            mode (bool, optional): _description_. Defaults to False.
            maxHands (int, optional): Max number of hands to look for. Defaults to 2.
            complexity (int, optional): 1 for higher detail, 0 for lower detail. Defaults to 1.
            detectionCon (float, optional): Ranges from 0-1. Defaults to .5.
            trackCon (float, optional): Ranges from 0-1. Defaults to .5.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.results = None
        
    def findHands(self, img, draw=True):
        """Identifies hands in image and stores results in class "result" variable

        Args:
            img (OpenCv image): Expect BGR image from openCV
            draw (bool, optional): Set to true to draw hand skeleton overlay over hand. Defaults to True.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                                                                                                       
    def findPosition(self, img, featureList, handNo=0, drawFeatures=True, drawBoundingBox = False):
        """Extracts features from results of findHands call and optionally draws them on image. 
           Returns list of feature positions and bounding box coordinates.

        Args:
            img (OpenCV image): Expects BGR openCV image
            featureList (int list): Include feature numbers you want drawn
            handNo (int, optional): Which hand to analyze. Defaults to 0.
            drawFeatures (bool, optional): Set to True to draw selected features from featureList onto image. Defaults to True.
            drawBoundingBox (bool, optional): Set to True to draw a bounding box around hand on image. Defaults to False.

        Returns:
            (int list, int list): (landmarkPositions, boundingBox) 
                                  landmarkPositions is of type [idNum, cx, cy] 
                                  
                                  bounding box is [xmin,ymin,xmax,ymax]
        """
        lmList = []
        bbox = []
        featureSet = set(featureList)
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if id in featureSet:
                        if drawFeatures:
                            cv2.circle(img,(cx,cy), 10, (255,255,255), cv2.FILLED)
        if lmList:
            xmin, ymin = min(lmList,key=lambda x: x[1])[1], min(lmList,key=lambda x: x[2])[2]
            xmax, ymax = max(lmList,key=lambda x: x[1])[1], max(lmList,key=lambda x: x[2])[2]
            bbox = [xmin,ymin,xmax,ymax]
            if drawBoundingBox:
                cv2.rectangle(img,(xmin-20,ymin-20),(xmax+20,ymax+20),(0,0,255),2)
        return lmList, bbox
    def findPositionAdvanced(self, img, featureList, handNo=0, drawFeatures=True, drawBoundingBox = False):
        """Extracts features from results of findHands call and optionally draws them on image. 
           Returns list of feature positions and bounding box coordinates.

        Args:
            img (OpenCV image): Expects BGR openCV image
            featureList (int list): Include feature numbers you want drawn
            handNo (int, optional): Which hand to analyze. Defaults to 0.
            drawFeatures (bool, optional): Set to True to draw selected features from featureList onto image. Defaults to True.
            drawBoundingBox (bool, optional): Set to True to draw a bounding box around hand on image. Defaults to False.

        Returns:
            (int list, int list): (landmarkPositions, boundingBox) 
                                  landmarkPositions is of type [idNum, cx, cy] 
                                  
                                  bounding box is [xmin,ymin,xmax,ymax]
        """
        if len(self.results.multi_hand_landmarks) - 1 < handNo:
            return None,None
        lmList = []
        bbox = []
        featureSet = set(featureList)
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy,lm.x,lm.y,lm.z])
                    if id in featureSet:
                        if drawFeatures:
                            cv2.circle(img,(cx,cy), 10, (255,255,255), cv2.FILLED)
        if lmList:
            xmin, ymin = min(lmList,key=lambda x: x[1])[1], min(lmList,key=lambda x: x[2])[2]
            xmax, ymax = max(lmList,key=lambda x: x[1])[1], max(lmList,key=lambda x: x[2])[2]
            bbox = [xmin,ymin,xmax,ymax]
            if drawBoundingBox:
                cv2.rectangle(img,(xmin-20,ymin-20),(xmax+20,ymax+20),(0,0,255),2)
        return lmList, bbox
       
       
       
            
def main():
    prevTime = 0
    currTime = 0 
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    

    # save
    with open('/Users/mahuja/Projects/HandRecognition/GestureControls/model.pkl', 'rb') as f:
        clf = pickle.load(f)
        
    while True:
        success, img = cap.read()
        if not success:
            continue
        detector.findHands(img)
        positions,bbox = detector.findPosition(img,[0,15,12],drawBoundingBox=True)
        # if positions:
        #     print(positions[3])
        
        if detector.results.multi_hand_landmarks:
            lms, bbox = detector.findPositionAdvanced(img,[],drawBoundingBox=False,drawFeatures=False)
            if lms:
                featRawCoords = []
                featRawCoordsNorm = []
                featConvertedCoords = []
                featConvertedCoordsNorm = []
                xmin, ymin = min(lms,key=lambda x: x[3])[3], min(lms,key=lambda x: x[4])[4]
                xmax, ymax = max(lms,key=lambda x: x[3])[3], max(lms,key=lambda x: x[4])[4]
                for i,lm in enumerate(lms):

                    id,cx,cy,x,y,z = lm

                    #featRawCoords.extend((x,y,z))
                    newX = np.interp(x, (xmin, xmax), (0, 1))
                    newY = np.interp(y, (ymin, ymax), (0, 1))
                    featRawCoords.extend((newX,newY,z))
                    featRawCoordsNorm.extend((newX,newY,z))
            
            predicted = clf.predict(np.array([featRawCoordsNorm]))
            
            cv2.putText(img,str(int(predicted)), (10,140),cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        
        currTime = time.time()
        fps = 1 / (currTime-prevTime)
        prevTime = currTime
        
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
        cv2.imshow("Image",img)
        #cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'): #waits one millisecond for input 'q', if detected we will stop capturing frames
            break
if __name__ == "__main__":
    main()