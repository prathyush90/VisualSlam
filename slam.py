import cv2
import pygame
import numpy as np
from scipy import ndimage
from frameprocessor import FrameProcessor
H = 1920//2
W = 1080//2
last = None
cap = cv2.VideoCapture('videos/test_countryroad.mp4')
orb = cv2.ORB_create(100)
processor = FrameProcessor(W, H)



while(cap.isOpened()):
    ret, frame = cap.read()


    if frame is None:
        break

    frame = cv2.resize(frame, (W, H))


    fmatch, matches, corners = processor.processFrame(frame)








    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(frame, (x, y), color=(0,0,255), radius=3)



    if(fmatch.any()):
        for pt1,pt2 in fmatch:
            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)


    cv2.imshow('Frame', frame)


    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
