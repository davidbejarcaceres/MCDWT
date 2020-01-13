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
from substract_offset_module import substract_offset



# Creates the command line arguments
parser = argparse.ArgumentParser("Calculates gain of an image calculating energies\nMore info: https://docs.opencv.org/3.1.0/d1/d5c/tutorial_py_kmeans_opencv.html")
parser.add_argument("-p" , help="Path to the MCDWT images: /tmp/video_transformed/16bits/", default = (os.path.join(tempDirNoSlash, "video_transformed", "16bit") + os.sep), type = str)
parser.add_argument("-n", help="Number of images (NOT SUB-BANDS): 20 bands = 5 images", default = 5, type = int)
parser.add_argument("-step", "--step", help="Quantization steps", default = 16, type = int)
parser.add_argument("-kmeans", "--kmeans", help="Activates the advanced quantizer with Kmeans (+CPU ussage)\Sometimes better results using less steps, better for sub-bands", action='store_true')                                

args = parser.parse_args() # Parses all the arguments
print("Starts")
K = args.step
nImages = args.n
path =  args.p
print("Path:  "+path)
bands = ["LL", "HL", "LH", "HH"]

if (args.kmeans):
    print("Using kmeans... ")
    for image in range(nImages):
        print("\nWorking in sub-bands from image: {:03d}".format(image))
        for band in bands:
            print("Band {}".format(band))
            subprocess.run("{} -O substract_offset.py -i {}{}{:03d}.png -o {}normalized_kmeans.png".format(pythonversion3OS, path, band , image, tempDir), shell=True, check=True)
            print("Offset Removed from the img to normalized")

            img = cv2.imread(f"{tempDir}normalized_kmeans.png", -1)
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
            if (band == "LL"):
                cv2.imshow('Quantized Sub-band LL K = {}'.format(K),res2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(f"{tempDir}quantizedKmeans.png", res2.astype(np.uint16)) # Saves the image to tmp indicating the number of steps
            subprocess.run("{} -O add_offset.py -i /tmp/quantizedKmeans.png -o {}{}{:03d}.png".format(pythonversion3OS, path, band , image), shell=True, check=True)
            print("Offset Added, sub-band ready!")
else:
    for image in range(nImages):
        print("\nWorking in sub-bands from image: {:03d}".format(image))
        for band in bands:
            print("Band {}".format(band))
            subprocess.run("{} -O substract_offset.py -i {}{}{:03d}.png -o {}normalized.png".format(pythonversion3OS, path, band , image, tempDir), shell=True, check=True)
            print("Offset Removed from the img to normalized")

            # Implements an uniform scalar [quantizer]
            imgTest = Image.open(f"{tempDir}normalized.png")
            im2Test = imgTest.quantize(K)
            if (band == "LL"):
                im2Test.show() # Displays the LL subbands
            im2Test.save(f"{tempDir}quantized.png","PNG")

            # Reverts to 16 bits images to work with MCDWT project REPLACING THE ORIGINAL IMAGE
            subprocess.run("{} -O add_offset.py -i {}quantized.png -o {}{}{:03d}.png".format(pythonversion3OS, tempDir, path, band , image), shell=True, check=True)
            print("Offset Added, sub-band ready!")
        
print("DONE!")    