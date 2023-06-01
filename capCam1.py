#!/usr/bin/env python3

import numpy as np
import cv2
import os
import time

timestr = time.strftime("%Y%m%d_%H%M%S")
print(timestr)
cap = cv2.VideoCapture('rtsp://admin:A12345678@192.168.43.191:554/cam/realmonitor?channel=1&subtype=0')


i=0
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == False:
		print("Capture Failed")
		break
	cwd = os.getcwd()	
	cv2.imwrite('/home/leaf/Documents/melProject/Cam1/C1mel'+timestr+'.jpg',frame)
	i+=1
	print("Capture Success")
	break

cap.release()
cv2.destroyAllWindows()


