import cv2
import numpy as np
import dlib
import winsound
import os
from imutils import face_utils
from gtts import gTTS

# emotion detection
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from playsound import playsound
# import vlc
face_classifier = cv2.CascadeClassifier(r'D:\Py_cbp\IIIT_Hack\Emotion_detection\face.xml')
classifier =load_model(r'D:\Py_cbp\IIIT_Hack\Emotion_detection\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

def emotionDetect():
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            if label=="Sad":
                playsound(r'D:\Py_cbp\IIIT_Hack\Emotion_detection\Nane.mp3')


        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
# Emotion detection

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frequency = 2500
duration = 2000

sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)

def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    face_frame = frame.copy()
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)


        left_blink = blinked(landmarks[36],landmarks[37], 
        	landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
        	landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        
        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            if(sleep>6):
                status="SLEEPING"
                os.system("cls")
                print(status)
                color = (255,0,0)
                winsound.Beep(frequency, duration)
            
        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status="Drowsy"
                os.system("cls")
                print(status)
                color = (0,0,255)
                winsound.Beep(frequency, duration)
        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status="Active"
                os.system("cls")
                print(status)
                emotionDetect()
                color = (0,255,0)
            
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # cv2.imshow("Frame", frame)
    # cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
      	break
