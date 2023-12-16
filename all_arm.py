# Before exe make sure to have installed mediapipe, if not >>
# pip install mediapipe
import mediapipe as mp
# Link for more info on pose estimation in the web of mediapipe
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
import cv2 as cv
import numpy as np
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import AngularServo
from time import sleep
import math

factory = PiGPIOFactory()

servo1 = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory = factory)
servo2 = AngularServo(23, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory = factory)

# LOCAL FUNCTIONS:
def calcular_angulo(a,b,c,d):
	a=np.array(a) # left shoulder
	b=np.array(b) # left elbow
	c=np.array(c) # left wrist
	d=np.array(d) # left hip
	# Calculate the input for each servo
	# angulo2 -> servo1 (first joint; references: hip, shoulder, elbow) 
	# angulo1 -> servo2 (second joint; references: shoulder, elbow, wrist)
	radianes1=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
	angulo1=np.abs(radianes1*180.0/np.pi)
	radianes2=np.arctan2(b[1]-a[1],b[0]-a[0])-np.arctan2(d[1]-a[1],d[0]-a[0])
	angulo2=np.abs(radianes2*180.0/np.pi)
	# We must make sure our values are ranged from 0 to 180 degrees
	if angulo1>180.0:
		angulo1=360-angulo1

	if (angulo2>180.0):
		angulo2=360-angulo2
    
	return angulo2, angulo1 
	
def recolorImage(frame,pose):
	# Each of the frames of the video must be transformed to RGB by recoloring
	img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
	img.flags.writeable = False
	# making a detection -> method process to store our detection in results var
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
#			.pose_conenctions gives us the different connections of different parts of our body (wrist, elbow and shoulder most importantly)
		# We use the method draw_landmarks to draw the points detected of our body in the video feed
		mp_draw.draw_landmarks(image,results_recolor.pose_landmarks, mp_pose.POSE_CONNECTIONS)
		# we store our landmarks to get the angles related to each of the body parts related to our robot arm joints
		landmarks = results_recolor.pose_landmarks.landmark
		# Each location is a vector of points [x,y] in a 2D system so we can proceed to get the angles
		shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
		elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
		wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
		hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
		# We get the angles we'll use as input for the servos 
		angle1, angle2 = calcular_angulo(shoulder,elbow,wrist,hip)
		# We must convert according to our specificacions 
		if(angle1 <= 180): # servo1 range it's (-90, 90)
			servo1.angle = (angle1 - 90)
		else:
			servo1.angle = (180 - 90)
		if(angle2 <= 140 ): # servo2 range it's (-90,90) 
			servo2.angle = ((angle2)- 90)
		else:
			servo2.angle = ((140) - 90)
		

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
#----MAIN-------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------
# variable to visualize our poses
mp_draw = mp.solutions.drawing_utils
# imports our pose estimation model
mp_pose = mp.solutions.pose
# get the camera feed
cam = cv.VideoCapture(0) 
mdc = 0.5 # parameter of minimal detection confidence(accuracy)
mtc = 0.5 # parameter of tracking accuracy

with mp_pose.Pose(min_detection_confidence=mdc, min_tracking_confidence=mtc) as pose:
	# Camera activation 
	readingCamera(cam,pose)
