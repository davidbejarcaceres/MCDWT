#!/usr/bin/env python3.6
from __future__ import print_function
import numpy as np
import cv2 as cv
import argparse

videoName = "shibuya.mp4" # A lot of movement
videoName2 = "vtest.avi" # Slow movent 

parser = argparse.ArgumentParser(description='This program shows how to use Optical Flow provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--shadows', action='store_true', help='When passed shadow detection is activated.')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(detectShadows= args.shadows) # Gaussian Mixture-based Background/Foreground Segmentation Algorithm.
else:
    backSub = cv.createBackgroundSubtractorKNN(detectShadows= args.shadows) # K-nearest neighbours - based Background/Foreground Segmentation


videoCaptured = cv.VideoCapture(args.input)
if not videoCaptured.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

print("Open CV:  "  + cv.__version__)
print("Working with video:  " + args.input)
print(f'Background substracytor method:  {args.algo},  Shadows= {args.shadows}')


while True:
    ret, frame = videoCaptured.read()
    if frame is None:
        break

    foreground = backSub.apply(frame)
    cv.imshow("Frame: ", frame)
    cv.imshow("Foreground Moving: ", foreground)

    keyboard = cv.waitKey(30);
    if keyboard == "q" or keyboard == 27:
        break


videoCaptured.release()
cv.destroyAllWindows()

print("\n Finished! \n")