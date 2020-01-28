# Developed by Davud Bejar caceres
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

N_threads: int = os.cpu_count()
cv.setNumThreads(N_threads)

opticalFlowGPUCalculator = cv.cuda_FarnebackOpticalFlow.create(10, 0.5, False, 15, 3, 5, 1.2, 0)

opticalFlowDual_TVL1Calculator = cv.cuda_OpticalFlowDual_TVL1.create( 0.25,  0.25,  0.3,  5,  5,  0.01,  30,  0.8,  0.0,  False )


# int 	numLevels = 5,
# double 	pyrScale = 0.5,
# bool 	fastPyramids = false,
# int 	winSize = 13,
# int 	numIters = 10,
# int 	polyN = 5,
# double 	polySigma = 1.1,
# int 	flags = 0 

# image1Path = os.path.join(thisPath,"basketball1.png")
# image2Path = os.path.join(thisPath,"basketball2.png")

image1Path = os.path.join(thisPath,"1.png")
image2Path = os.path.join(thisPath,"2.png")

def main():
    parser = argparse.ArgumentParser(description = "Returns the Dense Optical Flow calculated on GPU by OpenCV and shows the flow color encoded\n\n"
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

    flow = opticalFlowCuda(prevgray, gray)
    flowDualTV1 = opticalFlowCuda_Dual_TVL1(prevgray, gray)

    if args.view:
        colorFlow = OpticalFlowToColor.flow_to_color(flow, convert_to_bgr=False)
        colorFlowTVL1 = OpticalFlowToColor.flow_to_color(flowDualTV1, convert_to_bgr=False)
        cv.imshow('Optical Flow Color encoded on GPU', colorFlow)
        cv.waitKey(0)
        cv.destroyAllWindows()

        cv.imshow('Optical Flow TVL1 Color encoded on GPU', colorFlowTVL1)
        cv.waitKey(0)
        cv.destroyAllWindows()

    print("End")


def opticalFlowCuda(imgPrev: np.uint8, gray: np.uint8):
    ############  CUDA Optical Flow ###############
    g_prev_gpu = cv.cuda_GpuMat(imgPrev) # Uploads image to GPU
    g_next_gpu = cv.cuda_GpuMat(gray) # Uploads image to GPU
    flowGPU = opticalFlowGPUCalculator.calc(g_prev_gpu, g_next_gpu, None)  # Calculate on GPU
    flow = flowGPU.download() # Copies the optical flow from GPU to Host
    ###############################################
    return flow

def opticalFlowCuda_Dual_TVL1(imgPrev: np.uint8, gray: np.uint8):
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