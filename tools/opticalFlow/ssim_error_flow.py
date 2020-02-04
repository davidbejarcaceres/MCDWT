# Developed by Davud Bejar caceres
# Based on OpenCV C++ implementation: https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html

import numpy as np 
import cv2 as cv
import sys
import os
import argparse
import matplotlib.pyplot as plt
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
from readFlow2.readFlowFile import read as readFlow

cuda_enabled = False
try:
    cuMat1 = cv.cuda_GpuMat()
    opticalFlowGPUCalculator = cv.cuda_FarnebackOpticalFlow.create(10, 0.5, False, 15, 3, 5, 1.2, 0)
    cuda_enabled = True
except:
  print("No CUDA support")

# Benchmark parameters
N_threads: int = os.cpu_count()
cv.setNumThreads(N_threads)

opticalFlowGPUCalculator =  cv.cuda_FarnebackOpticalFlow.create(10, 0.5, False, 15, 3, 5, 1.2, 0) if cuda_enabled else None



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

    if frame1 is None:
        print("ERROR: File not found:  " + args.frame1)
        exit()

    if frame2 is None:
        print("ERROR: File not found:  " + args.frame2)
        exit()


    realFlow = readFlow(groundTruth)
    realFlowColor = computeImg(realFlow)
    print(realFlowColor.shape)

    flowFernerback = opticalFlowCuda(frame1, frame2) if  cuda_enabled else opticalFlowCPU(frame1, frame2)
    flowFernerbackColor = computeImg(flowFernerback)
    

    ssim_opencv = get_ssim_openCV(flowFernerbackColor, realFlowColor)
    error_MSE_numpy = np.square(np.subtract(flowFernerbackColor,realFlowColor)).mean()


    showFlowSSIM(frame1, realFlowColor, flowFernerbackColor, ssim_opencv, error_MSE_numpy);

    print("SSIM Error: " + str(ssim_opencv))

    return ssim_opencv;


def error_ssim_compareReal_Fernerback(frame1Path, frame2Path, realFlowPath, show = True):
    frame1 = cv.imread(frame1Path, cv.IMREAD_GRAYSCALE )
    frame2 = cv.imread(frame2Path, cv.IMREAD_GRAYSCALE )

    if frame1 is None:
        print("ERROR: File not found:  " + frame1)
        exit()

    if frame2 is None:
        print("ERROR: File not found:  " + frame2)
        exit()
    
    realFlow = readFlow(realFlowPath)
    realFlowColor = computeImg(realFlow)

    flowFernerback = opticalFlowCuda(frame1, frame2) if  cuda_enabled else opticalFlowCPU(frame1, frame2)
    flowFernerbackColor = computeImg(flowFernerback)
    

    ssim_opencv = get_ssim_openCV(flowFernerbackColor, realFlowColor)

    error_MSE_numpy = np.square(np.subtract(flowFernerbackColor,realFlowColor)).mean()

    if show:
        showFlowSSIM(frame1, realFlowColor, flowFernerbackColor, ssim_opencv, error_MSE_numpy);

    return ssim_opencv;
    



def showPairImages(image1, image2, error):

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()


    ax[0].imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    # ax[0].set_xlabel(label.format(mse_none, ssim_none))
    ax[0].set_title('Flujo óptico real')

    ax[1].imshow(cv.cvtColor(image2, cv.COLOR_BGR2RGB))
    ax[1].set_xlabel( "Error Similaridad Estrutal:  " + str(error) )
    ax[1].set_title('Flujo óptico conseguido')

    plt.tight_layout()
    plt.show()

    return 0;

def showFlowSSIM(image1, realFlow, flowFernerbackColor, ssim_error, sme_error):

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()


    ax[0].imshow(cv.cvtColor(image1, cv.COLOR_BGR2RGB))
    ax[0].set_title('Imagen de entrada 1 original')

    ax[1].imshow(cv.cvtColor(realFlow, cv.COLOR_BGR2RGB))
    ax[1].set_title('Flujo óptico real')

    ax[2].imshow(cv.cvtColor(flowFernerbackColor, cv.COLOR_BGR2RGB))
    ax[2].set_xlabel( "Error Similaridad Estrutal:  " + str(ssim_error) + "\n" + "Error Medio Cuadrado: " + str(np.around(sme_error, 3)) )
    ax[2].set_title('Flujo óptico conseguido')

    plt.tight_layout()
    plt.show()

    return 0;

# https://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
def get_sme_openCV(image1, image2):
    s1 = cv.absdiff(image1, image2);          
    s1 = s1 * s1    

    s = cv.sumElems(s1)   

    sse = s[0] + s[1] + s[2];

    if (sse <= 1e-10):
        return 0;
    else:
        mse =sse /(image1.channels() * image1.size);
        return mse;



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

def opticalFlowCPU(imgPrev: np.uint8, gray: np.uint8):
    return cv.calcOpticalFlowFarneback(imgPrev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def makeColorwheel():

	#  color encoding scheme

	#   adapted from the color circle idea described at
	#   http://members.shaw.ca/quadibloc/other/colint.htm

	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3]) # r g b

	col = 0
	#RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
	col += RY

	#YG
	colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
	colorwheel[col:YG+col, 1] = 255;
	col += YG;

	#GC
	colorwheel[col:GC+col, 1]= 255 
	colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
	col += GC;

	#CB
	colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
	colorwheel[col:CB+col, 2] = 255
	col += CB;

	#BM
	colorwheel[col:BM+col, 2]= 255 
	colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
	col += BM;

	#MR
	colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
	colorwheel[col:MR+col, 0] = 255
	return 	colorwheel

def computeColor(u, v):

	colorwheel = makeColorwheel();
	nan_u = np.isnan(u)
	nan_v = np.isnan(v)
	nan_u = np.where(nan_u)
	nan_v = np.where(nan_v) 

	u[nan_u] = 0
	u[nan_v] = 0
	v[nan_u] = 0 
	v[nan_v] = 0

	ncols = colorwheel.shape[0]
	radius = np.sqrt(u**2 + v**2)
	a = np.arctan2(-v, -u) / np.pi
	fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
	k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
	k1 = k0+1;
	k1[k1 == ncols] = 0
	f = fk - k0

	img = np.empty([k1.shape[0], k1.shape[1],3])
	ncolors = colorwheel.shape[1]
	for i in range(ncolors):
		tmp = colorwheel[:,i]
		col0 = tmp[k0]/255
		col1 = tmp[k1]/255
		col = (1-f)*col0 + f*col1
		idx = radius <= 1
		col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
		col[~idx] *= 0.75 # out of range
		img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

	return img.astype(np.uint8)


def computeImg(flow):

	eps = sys.float_info.epsilon
	UNKNOWN_FLOW_THRESH = 1e9
	UNKNOWN_FLOW = 1e10

	u = flow[: , : , 0]
	v = flow[: , : , 1]

	maxu = -999
	maxv = -999

	minu = 999
	minv = 999

	maxrad = -1
	#fix unknown flow
	greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
	greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
	u[greater_u] = 0
	u[greater_v] = 0
	v[greater_u] = 0 
	v[greater_v] = 0

	maxu = max([maxu, np.amax(u)])
	minu = min([minu, np.amin(u)])

	maxv = max([maxv, np.amax(v)])
	minv = min([minv, np.amin(v)])
	rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v)) 
	maxrad = max([maxrad, np.amax(rad)])
	print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

	u = u/(maxrad+eps)
	v = v/(maxrad+eps)
	img = computeColor(u, v)
	return img

TAG_STRING = 'PIEH'

def writeFlow(flow, filename):

	assert type(filename) is str, "file is not str %r" % str(filename)
	assert filename[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]

	height, width, nBands = flow.shape
	assert nBands == 2, "Number of bands = %r != 2" % nBands
	u = flow[: , : , 0]
	v = flow[: , : , 1]	
	assert u.shape == v.shape, "Invalid flow shape"
	height, width = u.shape

	f = open(filename,'wb')
	f.write(TAG_STRING)
	np.array(width).astype(np.int32).tofile(f)
	np.array(height).astype(np.int32).tofile(f)
	tmp = np.zeros((height, width*nBands))
	tmp[:,np.arange(width)*2] = u
	tmp[:,np.arange(width)*2 + 1] = v
	tmp.astype(np.float32).tofile(f)

	f.close()

if __name__ == "__main__":
    main()
    pass