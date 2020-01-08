#!/usr/bin/env python3.6

'''
example to show optical flow
USAGE: opt_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
Keys:
    ESC    - exit
'''

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



# import video


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)

def opticalFlowCuda(imgPrev, gray):
    ############  CUDA Optical Flow ###############
    g_prev_gpu = cv.cuda_GpuMat(imgPrev) # Uploads image to GPU
    g_next_gpu = cv.cuda_GpuMat(gray) # Uploads image to GPU
    flowGPU = opticalFlowGPUCalculator.calc(g_prev_gpu, g_next_gpu, None)  # Calculate on GPU
    flow = flowGPU.download() # Copies the optical flow from GPU to Host
    ###############################################
    return flow

def main():
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = cv.VideoCapture("vtest.avi")
    _ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    cur_glitch = prev.copy()
    # fps = int(cam.get(cv.CAP_PROP_FPS))

    opticalFlowGPUCalculator = cv.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
    frames = 0

    while(cam.isOpened()):
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        if cuda_enabled:
            flow = opticalFlowCuda(prevgray, gray)
            print("Frame GPU: " + str(frames))            
        else:
            flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            print("Frame CPU: " + str(frames))  

        
        prevgray = gray

        frames+=1

        cv.imshow('flow', draw_flow(gray, flow))
        draw_flow(gray, flow)
        if show_hsv:
            cv.imshow('flow HSV', draw_hsv(flow))
            draw_hsv(flow)
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()