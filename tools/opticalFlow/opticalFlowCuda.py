# Only for New Turin GPU CUDA with RayTracing https://devblogs.nvidia.com/opencv-optical-flow-algorithms-with-nvidia-turing-gpus/
import numpy as np 
import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)

frame1 = cv.imread( os.path.join(thisPath, 'basketball1.png'), cv.IMREAD_GRAYSCALE )
frame2 = cv.imread( os.path.join(thisPath, 'basketball2.png'), cv.IMREAD_GRAYSCALE )

#########################
imagen1 = cv.imread( os.path.join(thisPath, 'basketball1.png'))
imagen2 = cv.imread( os.path.join(thisPath, 'basketball2.png'))
frame1Gray = cv.cvtColor(imagen1, cv.COLOR_BGR2GRAY)
frame2Gray = cv.cvtColor(imagen2, cv.COLOR_BGR2GRAY)

npTmp = np.random.random((1024, 1024)).astype(np.float32)
npMat1 = np.stack([npTmp, npTmp], axis=2)
npMat2 = npMat1
cuMat1 = cv.cuda_GpuMat()
cuMat2 = cv.cuda_GpuMat()
cuMat3 = cv.cuda_GpuMat(frame1Gray) 
cuMat1.upload(frame1Gray)
cuMat2.upload(frame2Gray)

sumatoriaGPU = cv.cuda.add(cuMat1, cuMat2)
sumatoriaCPU = cv.add(frame1Gray, frame2Gray)

iguales = np.allclose(sumatoriaGPU.download(), sumatoriaCPU)

opticalFlowGPU = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
print(dir(opticalFlowGPU))

flowCUDA = opticalFlowGPU.calc(cuMat1, cuMat2, None)
flowCPU = cv.calcOpticalFlowFarneback(frame1Gray, frame2Gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flowCUDA_downloaded = flowCUDA.download()


igualesFlow = np.allclose(flowCUDA_downloaded, flowCPU)


#########################

nvof = cv.cuda_NvidiaOpticalFlow_1_0.create(frame1.shape[1], frame1.shape[0], 5, False, False, False, 1)
# nvof = cv.cuda.NvidiaOpticalFlow_1_0_create(frame1.shape[1],frame1.shape[0])

flow = nvof.calc(frame1, frame2, None)

flowUpSampled = nvof.upSampler(flow[0], frame1.shape[1], frame1.shape[0], nvof.getGridSize(), None)

cv.writeOpticalFlow('OpticalFlow.flo', flowUpSampled)

nvof.collectGarbage()