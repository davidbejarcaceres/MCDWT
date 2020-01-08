import time
import numpy as np
import cv2 as cv

npTmp = np.random.random((1024, 1024)).astype(np.float32)
npMat1 = np.stack([npTmp, npTmp], axis=2)
npMat2 = npMat1
cuMat1 = cv.cuda_GpuMat()
cuMat2 = cv.cuda_GpuMat()
cuMat1.upload(npMat1)
cuMat2.upload(npMat2)

start = time.time()
cv.cuda.gemm(cuMat1, cuMat2, 1, None, 0, None, 1)
end = time.time() - start
print(f'Time GPU CUDA: {end}')

startCPU = time.time()
cv.gemm(npMat1, npMat2, 1, None, 0, None, 1)
endCPU = time.time() - startCPU
print(f'Time CPU: {endCPU}')

gpuPerformance = round((endCPU / end), 2)
print(f'GPU is: {gpuPerformance}x faster')
