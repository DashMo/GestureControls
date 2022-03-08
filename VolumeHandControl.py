import cv2
import numpy as np
import time
import HandTrackingModule as hmd

wCam, hCam = 640, 480

detector = hmd.handDetector()

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

prevTime = 0
currTime = 0 

while True:
    success,img = cap.read()
    
    detector.findHands(img)
    detector.findPosition(img,[0,1,2])
    
    currTime = time.time()
    fps = 1 / (currTime-prevTime)
    prevTime = currTime
    
    cv2.putText(img,str(int(fps)),(40,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    
    cv2.imshow("Image",img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break