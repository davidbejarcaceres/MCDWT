import numpy as np
import cv2

cuda_enabled = False
cuda_turing_aceleration_sdk = False

try:
    cuMat1 = cv2.cuda_GpuMat()
    opticalFlowGPUCalculator = cv2.cuda_FarnebackOpticalFlow.create(
        10, 0.5, False, 15, 3, 5, 1.2, 0)
    cuda_enabled = True
except:
    print("No CUDA support")

if cuda_enabled:
    try:
        blank_image = np.zeros((500, 500, 3), np.uint8)
        blank_image.fill(200)
        nvof = cv2.cuda_NvidiaOpticalFlow_1_0.create(
            blank_image.shape[1], blank_image.shape[0], 5, False, False, False, 1)
        flow = nvof.calc(cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(
            blank_image, cv2.COLOR_BGR2GRAY), None)
        cuda_turing_aceleration_sdk = True
    except:
        print("No CUDA Turing GPU")
    pass


def generate_prediction(curr, next, base):
    flow = motion_estimation(curr, next)
    return estimate_frame(base, flow)


def motion_estimation(curr, next):
    curr_y = curr[:, :, 0]
    next_y = next[:, :, 0]

    return cv2.calcOpticalFlowFarneback(next_y, curr_y, None, 0.5, 3, 15, 3, 5, 1.2, 0) if cuda_enabled is False else opticalFlowCuda(next_y, curr_y)


def estimate_frame(base, flow):
    height, width = flow.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')

    return cv2.remap(base, map_xy, None,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def opticalFlowCuda(next_y: np.uint8, curr_y: np.uint8) -> cv2.UMat:
    if cuda_turing_aceleration_sdk:
        return opticalFlowCuda_testla(next_y, curr_y)
    ############  CUDA Optical Flow ###############
    next_y_gpu: cv2.cuda_GpuMat = cv.cuda_GpuMat()  # Allocates memory on GPU
    curr_y_gpu: cv2.cuda_GpuMat = cv.cuda_GpuMat()  # Allocates memory on GPU

    next_y_gpu.upload(next_y)  # Moves image to GPU allocated memory
    curr_y_gpu.upload(curr_y_gpu)  # Moves image to GPU allocated memory

    flowGPU: cv2.cuda_GpuMat = opticalFlowGPUCalculator.calc(
        next_y_gpu, curr_y_gpu, None)
    # Copies the optical flow from GPU memory to Host memory
    flow:  cv2.UMat = flowGPU.download()
    ###############################################
    return flow


def opticalFlowCuda_testla(next_y: np.uint8, curr_y: np.uint8) -> cv2.UMat:
    nvof = cv2.cuda_NvidiaOpticalFlow_1_0.create(
        next_y.shape[1], next_y.shape[0], 5, False, False, False, 1)
    flow:  cv2.UMat = nvof.calc(next_y, curr_y, None)
    return flow
