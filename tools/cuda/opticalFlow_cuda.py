# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import cv2 as cv
cv.setNumThreads(6)
cuda_enabled = False

try:
    cuMat1 = cv.cuda_GpuMat()
    cuda_enabled = True
    opticalFlowGPUCalculator = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
    pass
except:
    print("No CUDA Support, Using OpenCL")
    cuda_enabled = False
    exit()    

def opticalFlowCuda(imgPrev: np.uint8, gray: np.uint8):
    ############  CUDA Optical Flow ###############
    g_prev_gpu = cv.cuda_GpuMat(imgPrev) # Uploads image to GPU
    g_next_gpu = cv.cuda_GpuMat(gray) # Uploads image to GPU
    flowGPU = opticalFlowGPUCalculator.calc(g_prev_gpu, g_next_gpu, None)  # Calculate on GPU
    flow = flowGPU.download() # Copies the optical flow from GPU to Host
    ###############################################
    return flow