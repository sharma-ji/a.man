# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
capture = cv2.VideoCapture('7.mp4')
background = cv2.createBackgroundSubtractorMOG2()
humans = cv2.CascadeClassifier('haarcascade_fullbody.xml')
upper = cv2.CascadeClassifier('haarcascade_upperbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
while True:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	ret,image = capture.read()
	#image = cv2.flip(image,1)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()
	mask = background.apply(image)
	#mask = cv2.erode(mask,None, iterations=1)
	#mask = cv2.dilate(mask,None, iterations=1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	people = humans.detectMultiScale(gray, 2, 2)
	for (x,y,w,h) in people:
		cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
	upper_body = upper.detectMultiScale(gray,2,2)
	for(x,y,w,h) in upper_body:
		cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(orig, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
 
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 1)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
	# show the output images
	cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)
	cv2.imshow("mask", mask)
	k = cv2.waitKey(33)
	if k == 27:
		break



# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()
