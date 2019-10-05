# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:22:21 2019

@author: MUKHESH
"""

import cv2
import pickle
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
face_feature=cv2.CascadeClassifier('cv2_data/haarcascade_frontalface_alt2.xml')
eye_feature=cv2.CascadeClassifier('cv2_data/haarcascade_eye.xml')
recogniser=cv2.face.LBPHFaceRecognizer_create()

recogniser.read(BASE_DIR+'/trainer.yml')
cap=cv2.VideoCapture(0)

with open('lable.pickle','rb') as f:
    lables=pickle.load(f)
    label={v:k for k,v in lables.items()}

while True:
    i,frame=cap.read()
    #frame1=cv2.resize(frame,(150,150))
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face=face_feature.detectMultiScale(grey,scaleFactor=1.5,minNeighbors=4)
    for (x,y,w,h) in face:
        roi=grey[y:y+h,x:x+w]
        id_,conf=recogniser.predict(roi)
        if conf>=60:
            color=(255,0,0)
            stride=2
            eye=eye_feature.detectMultiScale(roi)
            for (ex,ey,ew,eh) in eye:
                cv2.rectangle(frame,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(124,123,213),1)
            
            name=label[id_]
            font=cv2.FONT_HERSHEY_SIMPLEX
            color=(255,123,0)
            stroke=1
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,stride)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()    
cv2.destroyAllWindows() 