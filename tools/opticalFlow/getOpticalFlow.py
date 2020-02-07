# Developed by David Bejar caceres
# Based on OpenCV C++ implementation: https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html

import numpy as np 
import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
import OpticalFlowToColor
import argparse
import tempfile
tempPath = tempfile.gettempdir()
projectPathOS = os.getcwd()
sys.path.append(projectPathOS)
projectPathOS = projectPathOS.replace("/tools", "") if sys.platform == "linux" else projectPathOS.replace("\\tools", "")
sys.path.append(projectPathOS)


# Benchmark parameters
N_threads: int = os.cpu_count()
cv.setNumThreads(N_threads)


image1Path = os.path.join(thisPath, 'basketball1.png')
image2Path = os.path.join(thisPath, 'basketball2.png')

def main():

    parser = argparse.ArgumentParser(description = "Returns the Dense Optical Flow calculated on CPU by OpenCV and shows the flow color encoded\n\n"
                                 "Example:\n\n"
                                 f"  getOpticalFlow -i {image1Path} -j {image2Path} \n")

    parser.add_argument("-i", "--frame1",
                        help="Input image 1", default=image1Path) #"../sequences/stockholm/000"

    parser.add_argument("-j", "--frame2",
                        help="Input image 2", default=image2Path)

    parser.add_argument("-v", "--view", action='store_false', help="View the color encoded Optical Flow")

    args = parser.parse_args()

    prevgray = cv.imread( args.frame1, cv.IMREAD_GRAYSCALE )
    gray = cv.imread( args.frame2, cv.IMREAD_GRAYSCALE )

    if prevgray is None:
        print("ERROR: File not found:  " + args.frame1)
        exit()

    if gray is None:
        print("ERROR: File not found:  " + args.frame2)
        exit()

    flow = getOpticalFlow(prevgray, gray)

    if args.view:
        colorFlow = OpticalFlowToColor.flow_to_color(flow, convert_to_bgr=False)
        cv.imshow('OPtical Flow Color encoded', colorFlow)
        cv.waitKey(0)

    print("End")

def getOpticalFlow(frame1, frame2) -> int:
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow



def opticalFlowCuda_Dual_TVL1(imgPrev: np.uint8, gray: np.uint8):
    opticalFlowDual_TVL1Calculator = cv.cuda_OpticalFlowDual_TVL1.create( 0.25,  0.25,  0.3,  5,  5,  0.01,  30,  0.8,  0.0,  False )
    ############  CUDA Optical Flow ###############
    g_prev_gpu = cv.cuda_GpuMat(imgPrev) # Uploads image to GPU
    g_next_gpu = cv.cuda_GpuMat(gray) # Uploads image to GPU
    flowGPU = opticalFlowDual_TVL1Calculator.calc(g_prev_gpu, g_next_gpu, None)  # Calculate on GPU
    flow = flowGPU.download() # Copies the optical flow from GPU to Host
    ###############################################
    return flow



if __name__ == "__main__":
    main()
    pass