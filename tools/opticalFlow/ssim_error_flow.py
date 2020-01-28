# Developed by Davud Bejar caceres
# Based on OpenCV C++ implementation: https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html

import numpy as np 
import cv2 as cv
import sys
import os
import argparse
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
from readFlow2.readFlowFile import read as readFlow

# Benchmark parameters
N_threads: int = os.cpu_count()
cv.setNumThreads(N_threads)

opticalFlowGPUCalculator = cv.cuda_FarnebackOpticalFlow.create(10, 0.5, False, 15, 3, 5, 1.2, 0)


image1Path = os.path.join(thisPath, '1.png')
image2Path = os.path.join(thisPath, '2.png')
groundTruth = os.path.join(thisPath, '1_2.flo')

def main():
    parser = argparse.ArgumentParser(description = "Returns the ssim error of optical flow compared to the ground truth\n\n"
                                 "Example:\n\n"
                                 f"  ssim_error_Opencv -i {image1Path} -j {image2Path} \n")

    parser.add_argument("-i", "--frame1",
                        help="Input image 1", default=image1Path) #"../sequences/stockholm/000"

    parser.add_argument("-j", "--frame2",
                        help="Input image 2", default=image2Path)

    parser.add_argument("-t", "--truth",
                        help="Ground Truth optical flow", default=groundTruth)

    args = parser.parse_args()

    frame1 = cv.imread( args.frame1, cv.IMREAD_GRAYSCALE )
    frame2 = cv.imread( args.frame2, cv.IMREAD_GRAYSCALE )
    realFlow = readFlow(groundTruth)
    if frame1 is None:
        print("ERROR: File not found:  " + args.frame1)
        exit()

    if frame2 is None:
        print("ERROR: File not found:  " + args.frame2)
        exit()

    ssim_opencv = get_ssim_openCV(frame1, frame2)
    print("SSIM Error: " + str(ssim_opencv))
    

# Can only accept gray images shape H x W x 2
def get_ssim_openCV(frame1, frame2) -> int:
    cv.setNumThreads(os.cpu_count())
    C1 = 6.5025
    C2 = 58.5225

    I1: cv.UMat = frame1.astype(np.float32)
    I2: cv.UMat = frame2.astype(np.float32)

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

def opticalFlowCuda(imgPrev: np.uint8, gray: np.uint8):
    ############  CUDA Optical Flow ###############
    g_prev_gpu = cv.cuda_GpuMat(imgPrev) # Uploads image to GPU
    g_next_gpu = cv.cuda_GpuMat(gray) # Uploads image to GPU
    flowGPU = opticalFlowGPUCalculator.calc(g_prev_gpu, g_next_gpu, None)  # Calculate on GPU
    flow = flowGPU.download() # Copies the optical flow from GPU to Host
    ###############################################
    return flow

if __name__ == "__main__":
    main()
    pass