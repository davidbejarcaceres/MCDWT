# Only for New Turin GPU CUDA with RayTracing https://devblogs.nvidia.com/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/
import numpy as np 
import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)

frame1 = cv.imread( os.path.join(thisPath, 'basketball1.png'), cv.IMREAD_GRAYSCALE )
frame2 = cv.imread( os.path.join(thisPath, 'basketball2.png'), cv.IMREAD_GRAYSCALE )

nvof = cv.cuda_NvidiaOpticalFlow_1_0.create(frame1.shape[1], frame1.shape[0], 5, False, False, False, 1)

flow = nvof.calc(frame1, frame2, None)

flowUpSampled = nvof.upSampler(flow[0], frame1.shape[1], frame1.shape[0], nvof.getGridSize(), None)

cv.writeOpticalFlow('OpticalFlow.flo', flowUpSampled)

nvof.collectGarbage()