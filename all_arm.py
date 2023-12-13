# Before exe make sure to have installed mediapipe, if not >>
# pip install mediapipe
import mediapipe as mp
# Link for more info on pose estimation in the web of mediapipe
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
import cv2 as cv
import numpy as np
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Servo
from time import sleep
import math

factory = PiGPIOFactory()

servo1 = Servo(18, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory = factory)
servo2 = Servo(23, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory = factory)

# LOCAL FUNCTIONS:
def calcular_angulo(a,b,c,d):
	a=np.array(a) #Hombro
	b=np.array(b) #Codo
	c=np.array(c) #Muñeca
	d=np.array(d) #cadera
	
	radianes1=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
	angulo1=np.abs(radianes1*180.0/np.pi)
	radianes2=np.arctan2(b[1]-a[1],b[0]-a[0])-np.arctan2(d[1]-a[1],d[0]-a[0])
	angulo2=np.abs(radianes2*180.0/np.pi)
	
	if angulo1>180.0:
		angulo1=360-angulo1

	if (angulo2>180.0):
		angulo2=360-angulo2
    
	return angulo2, angulo1 
	
def recolorImage(frame,pose):
	
	# recolor to RGB
	img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
	img.flags.writeable = False
	# make detection -> process to store our detection in results
	results = pose.process(img)
	# recolor back to BGR
	img.flags.writeable = True
	img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

	return results, img
	
def readingCamera(cam,pose):
#------ Camera activation / desactivation ---------------
	angulo_anterior = 0
	while cam.isOpened():
		ret, frame = cam.read() 
		results_recolor, image = recolorImage(frame,pose)
# 		render the detections:
#			.pose_landmarks gives us the following info: x,y,z,visibility
#			this represents every individual points of our estimation model
#			.pose conenctions gives us the different connections of different parts of our body (wrist, elbow and shoulder most importantly)
		mp_draw.draw_landmarks(image,results_recolor.pose_landmarks, mp_pose.POSE_CONNECTIONS)
		# guardar en una variable nuestras landmarks
		landmarks = results_recolor.pose_landmarks.landmark
		
		shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
		elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
		wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
		hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
		
		#print (calcular_angulo(hombro, codo, muñeca, cadera))
		angle1, angle2 = calcular_angulo(shoulder,elbow,wrist,hip)
		servo1.value = math.sin(math.radians( angle1*2))
		servo2.value = math.sin(math.radians( angle2*2))
		# show the image 
		cv.imshow('Mediapipe Feed', image)
		k = cv.waitKey(1)
		if k != -1:
			break
	cam.release() 
	cv.destroyAllWindows()
	print ('Camera quitted')
	servo.stop()
	GPIO.cleanup()
#---------------------------------------------------------------------------------------------------------------------------------------------
#----MAIN-------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
# variable to visualize our poses
mp_draw = mp.solutions.drawing_utils
# imports our pose estimation model
mp_pose = mp.solutions.pose

cam = cv.VideoCapture(0) 
mdc = 0.5 # parameter of minimal detection confidence(accuracy)
mtc = 0.5 # parameter of tracking accuracy

with mp_pose.Pose(min_detection_confidence=mdc, min_tracking_confidence=mtc) as pose:
	# Camera activation / desactivation 
	readingCamera(cam,pose)
