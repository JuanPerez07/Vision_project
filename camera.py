
import cv2

cam = cv2.VideoCapture(1)

while True:
	ret, image = cam.read()
	cv2.imshow('Imagetest',image)
	k = cv2.waitKey(1)
	if k != -1:
		break
cv2.imwrite('/home/vil/testimage.jpg', image)
cam.release()
cv2.destroyAllWindows()
