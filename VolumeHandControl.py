import cv2
import numpy as np
import time
import VolumeControlModule as vcm
import HandTrackingModule as hmd
import math

wCam, hCam = 640, 480

detector = hmd.handDetector(detectionCon=.7)
volControl = vcm.VolumeControl(os='mac',loop=10)


cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

prevTime = 0
currTime = 0 

volSlope = 100 / 175
volInter = -volSlope * 25 

while True:
    
    success,img = cap.read()
    if not success:
        continue
    detector.findHands(img)
    landmarks,bbox = detector.findPosition(img,[4,8],drawBoundingBox=True) #list with each landmark and the corresponding positions 
    if landmarks:
        
        
        # Find distance between index and thumb
        thumb_x,thumb_y = landmarks[4][1],landmarks[4][2]
        index_x,index_y = landmarks[8][1],landmarks[8][2]
        cx,cy = (thumb_x+index_x)//2, (thumb_y+index_y)//2
        cv2.line(img,(thumb_x,thumb_y),(index_x,index_y),(0,0,255),3)
        cv2.circle(img,(cx,cy),10,(0,0,255),cv2.FILLED)
    
        length = math.hypot(index_x-thumb_x,index_y-thumb_y) 
        #print(length)
        length = np.clip(length,25,200)
        
        
        #Convert volume and reduce resolution to mkae smoother
        #lets try 25-200 (hand range) to 0-100 (volume range)
        vol = volSlope * length + volInter    # np.interp(length,[25,200],[0,100])
        
        # Check what fingers are up
        
        #set volume
        volControl.set_volume(vol)    
    
    
    currTime = time.time()
    fps = 1 / (currTime-prevTime)
    prevTime = currTime
    
    cv2.putText(img,str(int(fps)),(40,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    
    cv2.imshow("Image",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break