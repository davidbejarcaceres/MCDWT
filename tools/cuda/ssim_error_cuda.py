# Developed by Davud Bejar caceres
# Based on OpenCV C++ implementation: https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html

import numpy as np 
import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
cuda_enabled = False
import time as time
try:
    cuMat1 = cv.cuda_GpuMat()
    cuda_enabled = True
    pass
except:
    print("No CUDA Support, Using OpenCL")
    cuda_enabled = False
    exit()  

# Benchmark parameters
N_threads: int = os.cpu_count()
cv.setNumThreads(N_threads)
ITER = 100


image1Path = os.path.join(thisPath, 'basketball1.png')
image2Path = os.path.join(thisPath, 'basketball2.png')
imagen3Path = os.path.join(thisPath, 'parque.jpg')

def main():
    frame1 = cv.imread( image1Path, cv.IMREAD_GRAYSCALE )
    frame2 = cv.imread( image2Path, cv.IMREAD_GRAYSCALE )

    frameA = frame1.astype(np.float32)
    frameB = frame2.astype(np.float32)

    ssim_opencv = get_ssim_openCV_cuda(frameA, frameB)
    print(ssim_opencv)


    # start_time = time.time()
    # for i in range(ITER):
    #     ssim_opencv = get_ssim_openCV_cuda(frameA, frameB)
    # print("---GPU ssim %s seconds ---" % (time.time() - start_time))   



    # start_time = time.time()
    # for i in range(ITER):
    #     ssim_opencv = get_ssim_openCV(frameA, frameB)
    # print("---CPU ssim %s seconds ---" % (time.time() - start_time))  


    # start_time = time.time()
    # for i in range(ITER):
    #     ssim_opencv = gausianCPU(frameA, frameB)
    # print("--- CPU Gaussian %s seconds ---" % (time.time() - start_time))  


    # start_time = time.time()
    # ssim_opencv = gausianCPU(frameA, frameB)
    # print("--- GPU Gaussian %s seconds ---" % (time.time() - start_time))  


    print("End")


def gausianCPU(frame1, frame2):
    mu1 = cv.GaussianBlur(frame1, (11, 11), 1.5);
    mu2 = cv.GaussianBlur(frame2, (11, 11), 1.5);



def gausianCUDA(frame1, frame2): 
    cap1 = cv.cuda_GpuMat(frame1)
    cap2 = cv.cuda_GpuMat(frame2)
    for i in range(ITER):
        sigma1_2 = cv.cuda.createGaussianFilter(cv.CV_32F, -1, (11, 11), 1.5).apply(cap1)
        sigma1_2 = cv.cuda.createGaussianFilter(cv.CV_32F, -1, (11, 11), 1.5).apply(cap2)
    

def get_ssim_openCV_cuda(frame1, frame2) -> int:
    cv.setNumThreads(os.cpu_count())
    C1 = 6.5025
    C2 = 58.5225

    I1: cv.cuda_GpuMat = cv.cuda_GpuMat(frame1)
    I2: cv.cuda_GpuMat = cv.cuda_GpuMat(frame2)

    I2_2: cv.UMat = frame2 * frame2
    I1_2: cv.UMat   = frame1 * frame1
    I1_I2: cv.UMat  = frame1 * frame2


    mu1 = cv.cuda.createGaussianFilter(cv.CV_32F, -1, (11, 11), 1.5).apply(I1).download()
    mu2 = cv.cuda.createGaussianFilter(cv.CV_32F, -1, (11, 11), 1.5).apply(I2).download()

    mu1_2   : cv.UMat =   mu1 * mu1
    mu2_2   : cv.UMat =   mu2 * mu2
    mu1_mu2 : cv.UMat =   mu1 * mu2
    sigma1_2 = cv.cuda.createGaussianFilter(cv.CV_32F, -1, (11, 11), 1.5).apply(cv.cuda_GpuMat(I1_2)).download()
    sigma1_2 -= mu1_2;

    sigma2_2 = cv.cuda.createGaussianFilter(cv.CV_32F, -1, (11, 11), 1.5).apply(cv.cuda_GpuMat(I2_2)).download()
    sigma2_2 -= mu2_2
    sigma12 = cv.cuda.createGaussianFilter(cv.CV_32F, -1, (11, 11), 1.5).apply(cv.cuda_GpuMat(I1_I2)).download()
    sigma12 -= mu1_mu2

    ################### FORMULA ############################

    t1: cv.UMat
    t2: cv.UMat
    t3: cv.UMat

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2

    ssim_map: cv.UMat
    ssim_map = cv.divide(t3, t1)

    mssim = cv.mean( ssim_map )
    return round(mssim[0], 4)


# TODO: Only for benchmark, delete later
def get_ssim_openCV(frame1, frame2) -> int:
    cv.setNumThreads(os.cpu_count())
    C1 = 6.5025
    C2 = 58.5225

    I1: cv.UMat = frame1
    I2: cv.UMat = frame2

    I2_2: cv.UMat = I2 * I2
    I1_2: cv.UMat   = I1 * I1
    I1_I2: cv.UMat  = I1 * I2 
    
    mu1: cv.UMat 
    mu2: cv.UMat 
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5);
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5);

    mu1_2   : cv.UMat=   mu1 * mu1
    mu2_2   : cv.UMat=   mu2 * mu2
    mu1_mu2 : cv.UMat=   mu1 * mu2

    sigma1_2: cv.UMat
    sigma2_2: cv.UMat
    sigma12: cv.UMat

    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5);
    sigma1_2 -= mu1_2;

    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5);
    sigma2_2 -= mu2_2;

    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ################### FORMULA ############################
    t1: cv.UMat
    t2: cv.UMat
    t3: cv.UMat

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2

    ssim_map: cv.UMat
    ssim_map = cv.divide(t3, t1)

    mssim = cv.mean( ssim_map )
    return round(mssim[0], 4)

if __name__ == "__main__":
    main()
    pass