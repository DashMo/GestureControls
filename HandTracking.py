import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
 
mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    #img.flags.writeable=False
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                if id == 15:
                    cv2.circle(img,(cx,cy), 10, (255,255,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    currTime = time.time()
    fps = 1 / (currTime-prevTime)
    prevTime = currTime
    
    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
    cv2.imshow("Image",img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'): #waits one millisecond for input 'q', if detected we will stop capturing frames
        break