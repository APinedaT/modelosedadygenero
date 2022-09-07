#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import datetime
import glob
from cupshelpers import Printer
import os
import shutil



def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes



faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
faceNet=cv2.dnn.readNet(faceModel,faceProto)
#ageNet=cv2.dnn.readNet(Model,modelProto)
dataset_folder_name = '/home/consultant1/Documents/Personal/ProyectoDeGrado/UTKface_inthewild/archive/utkface_aligned_cropped/UTKFace/*'
folder = '/home/consultant1/Documents/Personal/ProyectoDeGrado/UTKface_inthewild/archive/img_wrong'

files = glob.glob(dataset_folder_name)

for img in files:
    frame = cv2.imread(img) 
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if faceBoxes:
        #print("img:",img)
        #print("Is a face")
        pass
        #print("faceBoxes",faceBoxes)
    else:
        print("img:",img)
        print("Is not face")
        name = img.split("/")[10]
        shutil.move(img, folder+'/'+name)