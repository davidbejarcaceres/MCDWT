#!/usr/bin/env python3

# Note: swap the above line with the following two ones to switch
# between the standard and the optimized running mode.

#!/bin/sh
''''exec python3 -O -- "$0" ${1+"$@"} # '''

"""quantizer.py: Scalar Image quantizer for subbands in 16 bits using, also capable of using
                 klustering Kmeans (sometimes better results with less levels using more CPU)
                 Better for subbands."""

import cv2
import numpy as np
import math
import argparse
import subprocess
from cv2 import Sobel
from skimage import data, draw, transform, util, color, filters
from PIL import Image
import os
import tempfile
import sys
tempDir = tempfile.gettempdir() + os.sep
tempDirNoSlash = tempfile.gettempdir()
pythonversion3OS = "python3.6" if sys.platform == "linux" else "python"

# Creates the command line arguments
parser = argparse.ArgumentParser("Calculates gain of an image calculating energies\nMore info: https://docs.opencv.org/3.1.0/d1/d5c/tutorial_py_kmeans_opencv.html")
parser.add_argument("-i", help="Input image: /tmp/HH000.png", default = f"{tempDir}HH000.png")
parser.add_argument("-o", help="Output image: /tmp/quantized.png", default = f"{tempDir}quantized.png")
parser.add_argument("-step", "--step", help="Quantization steps", default = 16, type = int)
parser.add_argument("-kmeans", "--kmeans", help="Activates the advanced quantizer with Kmeans (+CPU ussage)\Sometimes better results using less steps, better for sub-bands", action='store_true')                                

args = parser.parse_args() # Parses all the arguments
K = args.step
subprocess.run("python -O substract_offset.py -i {} -o /tmp/normalized.png".format(args.i), shell=True, check=True)

if(args.kmeans):
    print("Using Kmeans ...")
    img = cv2.imread(args.i, -1)
    imgNormalized = cv2.imread(f"{tempDir}normalized.png")
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    # Shows the quantized image compared to the original.
    cv2.imshow('Quantized Image K = {}'.format(K),res2)
    cv2.imshow('Original Image',imgNormalized+128)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(("{}quantizedKmeans_K{}.png".format(tempDir, K)), res2.astype(np.uint16)) # Saves the image to tmp indicating the number of steps
else:
    print("Default method...")

    img = cv2.imread(f"{tempDir}normalized.png")
    #img += 128
    #cv2.imwrite("/tmp/normalized.png", img)
    # Light quantizer not using any improvement
    imgTest = Image.open(f"{tempDir}normalized.png")
    im2Test = imgTest.quantize(K)
    im2Test.show()
    im2Test.save("{}quantized_K{}.png".format(tempDir, K),"PNG")

    imgOriginal = cv2.imread(f"{tempDir}normalized.png")
    cv2.imshow('Original Image',imgOriginal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Reverts to 16 bits images to work with MCDWT project REPLACING THE ORIGINAL IMAGE
    #subprocess.run("python3 -O add_offset.py -i {} -o {}".format(args.i, args.i), shell=True, check=True)


