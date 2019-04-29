import cv2,os
import numpy as np
from PIL import Image 
import pickle
import urllib
import pyttsx
import speech_recognition as sr

count=0
r=sr.Recognizer()
engine = pyttsx.init()
recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'
url='http://192.168.0.5:8080/shot.jpg'
#cam = cv2.VideoCapture(0)
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1) #Creates a font
#font = cv2.FONT_HERSHEY_SIMPLEX
#fontscale = 1
#fontcolor = (255, 0, 0)
while True:
	imgResp = urllib.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        img = cv2.imdecode(imgNp,-1)
        cv2.imshow('IPWebcam',img)
 #   ret, im =cam.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in faces:
          nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
          cv2.rectangle(img,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
          if(conf<70):
        	if(nbr_predicted==1):
             		nbr_predicted='Akshay'
			print nbr_predicted
			count+=1
			print count
			if(count==5):
				
				#engine.runAndWait()
				count=0
				with sr.Microphone() as source:
					audio=r.listen(source)
				try:
					x=r.recognize_google(audio)
					if(x=='yes'):
						print("txt:"+r.recognize_google(audio))
						engine = pyttsx.init()
						engine.say('hello Akshay')
						engine.say('opening door')
						engine.runAndWait()	
				except:
					pass;			
        	elif(nbr_predicted==2):
             		nbr_predicted='Kaushal'
			print nbr_predicted 
	  else:
	     nbr_predicted='UNKNOWN'
	     print nbr_predicted 
       # cv2.cv.PutText(cv2.cv.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
	       
	cv2.imshow('img',img)
        cv2.waitKey(10)









