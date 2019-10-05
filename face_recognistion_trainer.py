# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:57:28 2019

@author: MUKHESH
"""

import cv2
import numpy as np
import os
from PIL import Image 
import pickle


face_feature=cv2.CascadeClassifier('cv2_data/haarcascade_frontalface_alt2.xml')
recogniser=cv2.face.LBPHFaceRecognizer_create()
Y_train={}

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,'images')
X_train=[]
Y=[]
cur=1

for root,dirname,file in os.walk(image_dir):
    for files in file: 
        if files.endswith('jpg') or files.endswith('png') or files.endswith('JPG'):
            image=Image.open(root+'/'+files).convert('L')
            #image=image.resize((150,150), Image.ANTIALIAS)
            image_array=np.array(image,'uint8')
            label=os.path.basename(root).replace(' ','_').lower()
            if label not in Y_train:
                Y_train[label]=cur
                cur+=1
            id_=Y_train[label]
            face=face_feature.detectMultiScale(image_array)
            for (x,y,w,h) in face:
                roi=image_array[y:y+h,x:x+w]
                X_train.append(roi)
                Y.append(id_)
         
recogniser.train(X_train,np.array(Y))
with open('lable.pickle','wb') as f:
    pickle.dump(Y_train,f)

recogniser.save(BASE_DIR+'/trainer.yml')




