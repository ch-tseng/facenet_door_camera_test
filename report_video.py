# -*- coding: utf-8 -*-

import numpy as np
import os, time
import cv2
import h5py
import imutils
import dlib
from imutils.face_utils import rect_to_bb
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

valid = "report/valid/"
compare = "report/test/"
video_file = "door_come_in.mp4"
output_video = "output/video/door" + str(round(time.time(),0)) + ".avi"
face_detect = "mtcnn"

min_faceSzie = (40, 40)
cascade_path = 'haarcascade_frontalface_alt2.xml'
min_score = 0.90
image_size = 160
giveupScore = 0.8
black_padding_width = 0  #add padding width for the face area

make_dataset = False
load_dataset = True
dataset_file = "officedoor.h5"
tracker_type = "KCF"  #BOOSTING, CSRT, TLD, MIL, KCF, MEDIANFLOW, MOSSE

#pretrained Keras model (trained by MS-Celeb-1M dataset)
model_path = 'model/facenet_keras.h5'
model = load_model(model_path)

if(face_detect=="mtcnn"):
    detector = MTCNN()
elif(face_detect=="dlib"):
    detector = dlib.get_frontal_face_detector()
else:
    detector = cv2.CascadeClassifier(cascade_path)

#-----------------------------------------------------------------------------

def tracker(bbox, frame):
    trackerType =  cv2.Tracker_create(trackerType.upper())
    tracker = OPENCV_OBJECT_TRACKERS[trackerType]()

    ok = tracker.init(frame, bbox)


def prewhiten(x):
    #cv2.imshow("Before", x)
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj

    #cv2.imshow("After", y)
    #cv2.waitKey(0)
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def align_image(img, margin):
    faces = []
    if(face_detect=="mtcnn"):
        allfaces = detector.detect_faces(img)
        for face in allfaces:
            print("face", face["box"])
            x = face["box"][0]
            y = face["box"][1]
            w = face["box"][2]
            h = face["box"][3]
            faces.append((int(x),int(y),int(w),int(h)))

    elif(face_detect=="dlib"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            faces.append((int(x),int(y),int(w),int(h)))

    else:
        allfaces = detector.detectMultiScale(img, scaleFactor=1.10, minNeighbors=5)
        for face in allfaces:
            (x, y, w, h) = face
            faces.append((int(x),int(y),int(w),int(h)))


    if(len(faces)>0):
        imgFaces = []
        bboxes = []
        for face in faces:
            (x, y, w, h) = face
            if(w>min_faceSzie[0] and h>min_faceSzie[1]):
                #print("     w,h=",w,h)
                faceArea = img[y:y+h, x:x+w]
                w = faceArea.shape[1]
                h = faceArea.shape[0]
                faceMargin = np.zeros((h+margin*2, w+margin*2, 3), dtype = "uint8")
                faceMargin[margin:margin+h, margin:margin+w] = faceArea
                cv2.imwrite("tmp/"+str(time.time())+".jpg", faceMargin)
                #aligned = resize(faceMargin, (image_size, image_size), mode='reflect')
                aligned = cv2.resize(faceMargin ,(image_size, image_size))
                #cv2.imwrite("tmp/"+str(time.time())+"_aligned.jpg", aligned)
                imgFaces.append(aligned)
                bboxes.append((x, y, w, h))

        if(len(bboxes)>0):
            return imgFaces, bboxes
        else:
            return None, None

    else:
        return None, None

def preProcess(img):
    whitenImg = prewhiten(img)
    whitenImg = whitenImg[np.newaxis, :]
    return whitenImg

#-------------------------------------------------

def face2name(face, faceEMBS, faceNames):
    #print(len(faceEMBS), len(faceNames))
    imgFace = preProcess(face)
    embs = l2_normalize(np.concatenate(model.predict(imgFace)))

    smallist_id = 0
    smallist_embs = 999
    for id, valid in enumerate(faceEMBS):
        distanceNum = distance.euclidean(embs, valid)
        #if(distanceNum>giveupScore):
        #    smallist_embs = distanceNum
        #    smallist_id = id
        #    print(distanceNum, "--> give up")
            #break
        #else:
        #print("     ", faceNames[id].decode(), distanceNum)
        if(smallist_embs>distanceNum):
            smallist_embs = distanceNum
            smallist_id = id

    print(faceNames[smallist_id].decode(), smallist_embs)
    return smallist_id, faceNames[smallist_id].decode(), smallist_embs

def draw_text(bbox, txt, img):
    fontSize = round(img.shape[0] / 930, 1)
    if(fontSize<0.35): fontSize = 0.35
    boldNum = int(img.shape[0] / 500)
    if(boldNum<1): boldNum = 1

    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),boldNum)
    cv2.putText(img, txt, (x, y-(boldNum*3)), cv2.FONT_HERSHEY_COMPLEX, fontSize, (255,0,255), boldNum)

    return img

# ----------------------------------------------------

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

trackStatus = 0


valid_names = []
valid_embs = []

if(make_dataset == True and load_dataset == False):

    hf = h5py.File(dataset_file, 'w')

    for username in os.listdir(valid):
        print("     Load ", username)
        for img_file in os.listdir(valid + username):
            filename, file_extension = os.path.splitext(img_file)

            if(file_extension.upper() in (".JPG", "PNG", "JPEG", "BMP")):
                imgValid = cv2.imread(valid+username+"/"+img_file)
                print("     ", valid+username+"/"+img_file)
                aligned, _ = align_image(imgValid, black_padding_width)
                if(aligned is None):
                    print("     Cannot find any face in image: {}".format(valid+username+"/"+img_file))
                else:
                    faceImg = preProcess(aligned[0])
                    embs = l2_normalize(np.concatenate(model.predict(faceImg)))
                    name = username.split(" ")[0]
                    valid_names.append(name.encode())
                    valid_embs.append(embs)

    hf.create_dataset("names", data=np.array(valid_names))
    hf.create_dataset("embs", data=np.array(valid_embs))
    hf.close()

    print("HF file saved, valid names:", valid_names)

else:
    hf = h5py.File(dataset_file, 'r')
    valid_names = hf.get('names')
    valid_embs = hf.get('embs')

    print("HF file loaded, valid names:", valid_names)

if(len(valid_names)>0):
    print("Length:", len(valid_names))
    VIDEO_IN = cv2.VideoCapture(video_file)
    # Get current width of frame
    width = VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    # Get current height of frame
    height = VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video,fourcc, 30.0, (int(width),int(height)))

    i = 0
    #for folder in os.listdir(compare):
        #for img_file in os.listdir(compare+folder):
    hasFrame = True
    while hasFrame:
        hasFrame, imgCompared = VIDEO_IN.read()
        i += 1
        aligned, faceBoxes = align_image(imgCompared, 6)
        if(faceBoxes is not None):
            #if(len(faceBoxes)>0):
            for id, face in enumerate(faceBoxes):
                valid_id, valid_name, score = face2name(aligned[id], valid_embs, valid_names)
                if(score<min_score):
                    txtName = valid_name
                    ok = tracker.init(imgCompared, face)
                    
                    trackStatus = 1

                    while trackStatus==1 and hasFrame==True:
                        hasFrame, imgCompared = VIDEO_IN.read()
                        #frameDisplay = imgCompared.copy()
                        ok, face = tracker.update(imgCompared)
                        if ok and (face[2]>min_faceSzie[0] and face[3]>min_faceSzie[1]):
                            print("Tracking.....", valid_name + "(" + str(round(score,2)) + ")")
                            imgCompared = draw_text(face, valid_name + "(" + str(round(score,1)) + ")", imgCompared)
                        else:
                            print("Lost tracking.....")
                            trackStatus = 0

                        cv2.imshow("People", imutils.resize(imgCompared, width=640))
                        cv2.waitKey(1)
                        out.write(imgCompared)

                else:
                    txtName = "unknow"

                    imgCompared = draw_text(face, txtName, imgCompared)
                    cv2.imshow("People", imutils.resize(imgCompared, width=640))
                    out.write(imgCompared)
                    cv2.waitKey(1)

        else:
            txtName = "noface"
            cv2.putText(imgCompared, "No face", (80,50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,255,0), 2)
            out.write(imgCompared)
            cv2.imshow("People", imutils.resize(imgCompared, width=640))
            cv2.waitKey(1)

        print("frame: ", i)

        #saveFolder = "output/" + txtName
        #if not os.path.exists(saveFolder):
        #    os.makedirs(saveFolder)

        out.write(imgCompared)
        cv2.imshow("People", imutils.resize(imgCompared, width=640))
        cv2.waitKey(1)

else:
    print("There is no any face in valid images.")
