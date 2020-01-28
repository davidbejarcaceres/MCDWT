# Developed by: David Bejar Caceres
# Benchmark of ssim calculation:
# Pure OpenCV 7.72ms
# Scikit: 18.04ms
# tf_ssim TensorFlow Implementation 40.73ms
# Tensorflow api implementation 159ms

import numpy as np 
import cv2 as cv
import sys
import os
from time import time
import timeit
thisPath = sys.path[0]
# Import Methods to calculate the ssim error
from ssim_error_OpenCV import get_ssim_openCV
from ssim_error_tensorflow import tf_ssim
from ssim_error_tensorflow import get_ssim_tensorFlow_float32
from ssim_error_scikit import ssim_scikit

# Benchmark parameters
N_threads = os.cpu_count()
iterations = 100
cv.setNumThreads(N_threads)

image1Path = os.path.join(thisPath, 'basketball1.png')
image2Path = os.path.join(thisPath, 'basketball2.png')


def main():
    print("\n######################################")
    frame1_gray = cv.imread(image1Path, cv.IMREAD_GRAYSCALE)
    frame2_gray = cv.imread(image2Path, cv.IMREAD_GRAYSCALE)

    frame1_color = cv.imread(image1Path).astype(np.float32)
    frame2_color = cv.imread(image2Path).astype(np.float32)

    # t = timeit.timeit(img_cal(image, 'UMat'), globals=globals(), number=N)/N*1000
    start = time()
    for i in range(iterations):
        get_ssim_openCV(frame1_gray, frame2_gray)
    
    totaltime = time() - start
    processor = 'get_ssim_openCV Pure OpenCV from docs' 

    print('%s avg. with %d threads: %0.2f ms' % (processor, cv.getNumThreads(), totaltime))


    start = time()
    for i in range(iterations):
        tf_ssim(frame1_gray, frame2_gray)
    
    totaltime = time() - start
    processor = 'tf_ssim implementation using tensorflow not the native function' 

    print('%s avg. with %d threads: %0.2f ms' % (processor, cv.getNumThreads(), totaltime))




    start = time()
    for i in range(iterations):
        get_ssim_tensorFlow_float32(frame1_color, frame2_color)
    
    totaltime = time() - start
    processor = 'getSSIM Using the native TensorFlow API and float32' 

    print('%s avg. with %d threads: %0.2f ms' % (processor, cv.getNumThreads(), totaltime))





    start = time()
    for i in range(iterations):
        ssim_scikit(frame1_gray, frame2_gray)
    
    totaltime = time() - start
    processor = 'ssim_scikit' 

    print('%s avg. with %d threads: %0.2f ms' % (processor, cv.getNumThreads(), totaltime))

    print("\n######################################")


if __name__ == "__main__":
    main()
    pass