import cv2
import pandas
import numpy
import urllib.request
import os
from datetime import datetime
import face_recognition
import pandas as pd

# path=r'D:\python\attendance\attendance\image_folder'
# url='http://192.168.231.162/cam-hi.jpg'
#
# if 'Attendance.csv' in os.listdir(os.path.join(os.getcwd(),'attendance')):
#     print("there iss....")
#     os.remove("Attendance.csv")
# else:
#     df=pd.DataFrame(list())
#     df.to_csv("Attendance.csv")
#
# images=[]
# classNames=[]
# myList=os.listdir(path)
#
# print(myList)
#
# for cl in myList:
#     curImg=cv2.imread({path}/{cl})
#     images.append(curImg)
#     classNames.append(os.path.split(cl)[0])

imgElon=face_recognition.load_image_file("imageBasic/elon_mask.jpg")
imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("imageBasic/bill-gates-jpg.jpg")
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


############### Detecting Face Location ################################################################

face_loc=face_recognition.face_locations(imgElon)[0]
encodeElon=face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(face_loc[3],face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)
print(face_loc)

############## Detecting Face Location ##############################################################


face_locTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(face_locTest[3],face_locTest[0]),(face_locTest[1],face_locTest[2]),(255,0,255),2)
print(face_locTest)


########## Recognize Face #################################################################

result= face_recognition.compare_faces([encodeElon],encodeTest)
print(result)

########## Recognize Face ###################################################################
############ Face Distance ######################################################


faceDist=face_recognition.face_distance([encodeElon],encodeTest)
print(faceDist)

############ Face Distance ######################################################

cv2.putText(imgTest,f'{result}{faceDist[0]}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)






cv2.imshow("Elon Mask",imgElon)
cv2.imshow("Elon Test",imgTest)




cv2.waitKey(0)




