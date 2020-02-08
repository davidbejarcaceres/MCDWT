"""
cpu_gpu.py
An OpenCL-OpenCV-Python CPU vs GPU comparison
"""
import cv2
from time import time
import os
import numpy as np

import timeit
import sys

N_threads = os.cpu_count() 
iterations = 5
cv2.setNumThreads(N_threads)

# A simple image pipeline that runs on both Mat and Umat
def img_cal_UMAT_OPENCL(img, mode):
    img = cv2.UMat(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    img = cv2.Canny(img, 0, 50)
    flow = cv2.calcOpticalFlowFarneback(img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)        
    return cv2.UMat.get(img)

# A simple image pipeline that runs on both Mat and Umat
def img_cal_MAT_CPU(img, mode):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    img = cv2.Canny(img, 0, 50)
    flow = cv2.calcOpticalFlowFarneback(img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return img



# A simple image pipeline that runs on both Mat and Umat
# def img_calGPU(img, mode):
#     opticalFlowGPU = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     g_next_gpu = cv2.cuda_GpuMat(img)
#     g_prev_gpu = cv2.cuda_GpuMat(img)
#     # img = cv2.GaussianBlur(img, (7, 7), 1.5)
#     # img = cv2.Canny(img, 0, 50)
#     flowGPU = opticalFlowGPU.calc(g_prev_gpu, g_next_gpu, None)
#     img = flowGPU.download()
#     imgNext = g_next_gpu.download()
#     return img

# Timing function
def run(n_threads, N, image):
    # t = timeit.timeit(img_cal(image, 'UMat'), globals=globals(), number=N)/N*1000
    start = time()
    for i in range(iterations):
        img_cal_UMAT_OPENCL(image, 'UMat')
    
    totaltime = time() - start
    processor = 'UMat OpenCL' 

    print('%s avg. with %d threads: %0.2f ms' % (processor, cv2.getNumThreads(), totaltime))


    start = time()
    for i in range(iterations):
        img_cal_MAT_CPU(image, 'MatCPU')
    
    totaltime = time() - start
    processor = 'Mat CPU' 

    print('%s avg. with %d threads: %0.2f ms' % (processor, cv2.getNumThreads(), totaltime))




    # start = time()
    # for i in range(iterations):
    #     img_calGPU(image, 'MatCPU')
    
    # totaltime = time() - start
    # processor = 'CUDA GPU' 

    # print('%s avg. with %d threads: %0.2f ms' % (processor, cv2.getNumThreads(), totaltime))
    
img = np.zeros((4000, 4000, 3), np.uint8)
print(img.shape)
N = 1000
run(n_threads=N_threads, N=N, image=img)

# threads = [1,  16]

# processor = {'GPU': "img_cal(img_UMat)", 
#              'CPU': "img_cal(img)"}
# results = {}
# for n in range(N_threads): 
#     for pro in processor.keys():
#         results[pro,n] = run(processor=pro, 
#                              function= processor[pro], 
#                              n_threads=n, N=N,
#                              image=img)

# print('\nGPU speed increase over 1 CPU thread [%%]: %0.2f' % \
#       (results[('CPU', 1)]/results[('GPU', 1)]*100))
# print('CPU speed increase on 4 threads versus 1 thread [%%]: %0.2f' % \
#       (results[('CPU', 1)]/results[('CPU', 16)]*100))
# print('GPU speed increase versus 4 threads [%%]: %0.2f' % \
#       (results[('CPU', 4)]/results[('CPU', 1)]*100))