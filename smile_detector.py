# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:36:53 2019

@author: MUKHESH
"""

import cv2


face=cv2.CascadeClassifier('cv2_data/haarcascade_frontalface_default.xml')
eye=cv2.CascadeClassifier('cv2_data/haarcascade_eye.xml')
smile=cv2.CascadeClassifier('cv2_data/haarcascade_smile.xml')

cap=cv2.VideoCapture(0)

while True:
    
    _,frame=cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(grey,1.3,7)
    for (x,y,w,h) in faces:
        roi=grey[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(238,234,212),2)
        smiles=smile.detectMultiScale(roi,1.1,22)
        eyes=eye.detectMultiScale(roi,1.3,22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(102,134,212),2)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(frame,(x+sx,y+sy),(x+sx+sw,y+sy+sh),(245,134,212),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()