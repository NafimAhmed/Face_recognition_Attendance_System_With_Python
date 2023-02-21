import cv2
import numpy as np
import pandas
import numpy
import urllib.request
import os
from datetime import datetime
import face_recognition
import pandas as pd

path="ImagesAttendance"

images=[]
className=[]
mylist= os.listdir(path)
print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])

print(className)

def findEncodings(images):
    encodList=[]

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodList.append(encode)
    return encodList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readline()
        nameList = []
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M')
            f.writelines(f'\n{name},{dtString}')


#markAttendance('Elon')




encodeListKnown=findEncodings(images)

print(len(encodeListKnown))


cap= cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,.25,.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    for encodeface,faceLoc in zip(encodesCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDist=face_recognition.face_distance(encodeListKnown,encodeface)
        print(faceDist)
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:
            name=className[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


# face_locTest=face_recognition.face_locations(imgTest)[0]
# encodeTest=face_recognition.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(face_locTest[3],face_locTest[0]),(face_locTest[1],face_locTest[2]),(255,0,255),2)
# print(face_locTest)
#
#
# ########## Recognize Face #################################################################
#
# result= face_recognition.compare_faces([encodeElon],encodeTest)
# print(result)

########## Recognize Face ###################################################################
############ Face Distance ######################################################


# faceDist=face_recognition.face_distance([encodeElon],encodeTest)
# print(faceDist)

# imgElon=face_recognition.load_image_file("imageBasic/elon_mask.jpg")
# imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
# imgTest=face_recognition.load_image_file("imageBasic/bill-gates-jpg.jpg")
# imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
