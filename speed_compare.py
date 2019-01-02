from mtcnn.mtcnn import MTCNN
import cv2
import time
import imutils

faceimg = "faces10.jpg"

org = cv2.imread(faceimg)


img = org.copy()
start = time.time()
detector = MTCNN()
faces = detector.detect_faces(org)

for face in faces:
    print(face["box"])
    x = face["box"][0]
    y = face["box"][1]
    w = face["box"][2]
    h = face["box"][3]
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("test", imutils.resize(img, width=800))
print("MTCNN time: {}, found {} faces".format(time.time()-start, len(faces)))
cv2.waitKey(0)

#----------------------------------------------------
min_faceSzie = (30, 30)
cascade_path = 'haarcascade_frontalface_default.xml'

img = org.copy()
start = time.time()
cascade = cv2.CascadeClassifier(cascade_path)
faces = cascade.detectMultiScale(org, scaleFactor=1.1, minNeighbors=6)
for face in faces:
    (x, y, w, h) = face
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("test", imutils.resize(img, width=800))
print("Cascade time: {}, found {} faces".format(time.time()-start, len(faces)))
cv2.waitKey(0)

#----------------------------------------------------------

import dlib
from imutils.face_utils import rect_to_bb

img = org.copy()
detector = dlib.get_frontal_face_detector()

start = time.time()
gray = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 2)
for rect in rects:
    (x, y, w, h) = rect_to_bb(rect)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("test", imutils.resize(img, width=800))
print("Dlib time: {}, found {} faces".format(time.time()-start, len(faces)))
cv2.waitKey(0)

