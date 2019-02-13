#!/usr/bin/env python

import cv2
import numpy as np
import math
import argparse
from cv2 import Sobel
from skimage import data, draw, transform, util, color, filters
import pylab



class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

parser = argparse.ArgumentParser(description = "Displays information about an image\n\n"
                                 "Example:\n\n"
                                 "  python show_statistics.py -i ../sequences/stockholm/000\n",
                                 formatter_class=CustomFormatter)

parser.add_argument("-i", "--image",
                    help="Input image", default="/tmp/stockholm/000")

args = parser.parse_args()

def compute_entropy(d, counter):
    for x in d:
        d[x] /= 1.*counter

    en = 0.
    for x in d:
        en += d[x] * math.log(d[x])/math.log(2.0)

    return -en

image = cv2.imread(args.image, -1)
tmp = image.astype(np.float32)
tmp -= 32768.0
image = tmp.astype(np.int16)

width = image.shape[0]
height = image.shape[1]
number_of_pixels = width * height
components = image.shape[2]

# Calculates the energy
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
component_x = Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
component_y = Sobel(gray, cv2.CV_64F, 1, 0 , ksize = 3)
abs_component_x = cv2.convertScaleAbs(component_x)
abs_component_y = cv2.convertScaleAbs(component_y)
energyGris = cv2.addWeighted(abs_component_x, 0.5, abs_component_y, 0.5, 0)
print("Canales en componentes: {}".format(np.sum(energyGris)))
#pylab.imshow(energy)

energy = util.img_as_float(image)
energy = filters.sobel(color.rgb2gray(energy))
print("Sum Energy 1: {}".format(np.sum(energy)))
pylab.title("Energìa de la imagen")
pylab.imshow(energy), pylab.show()

energy2 = np.power(image, 2)
print("Sum Energy 2: {}".format(np.sum(energy2)))


histogram = [None]*components
for c in range(components):
    histogram[c] = {}

for y in range(width):
    for x in range(height):
        for c in range(components):
            val = image[y,x,c]
            if val not in histogram[c]:
                histogram[c][val] = 1
            else:
                histogram[c][val] += 1

entropy = [None]*components
for c in range(components):
    entropy[c] = compute_entropy(histogram[c], number_of_pixels)

max = [None] * components
min = [None] * components
dynamic_range = [None] * components
mean = [None] * components
for c in range(components):
    max[c] = np.amax(image[:,:,c])
    min[c] = np.amin(image[:,:,c])
    dynamic_range[c] = max[c] - min[c]
    mean[c] = np.mean(image[:,:,c])

print("Image: {}".format(args.image))
print("Width: {}".format(width))
print("Height: {}".format(height))
print("Components: {}".format(components))
print("Number of pixels: {}".format(number_of_pixels))
print("Energy: {}".format(np.sum(energy)))
for c in range(components):
    print("Max value of component {}: {}".format(c, max[c]))
for c in range(components):
    print("Min value of component {}: {}".format(c, min[c]))
for c in range(components):
    print("Dynamic range of component {}: {}".format(c, dynamic_range[c]))
for c in range(components):
    print("Mean of component {}: {}".format(c, mean[c]))
for c in range(components):
    print("Entropy of component {}: {}".format(c, entropy[c]))

indices = [None] * components
for c in range(components):
    print("Component {}".format(c))
    print("{0: <8} {1: <10} {2: <10}".format("position", "coordinates", "value"))
    # https://stackoverflow.com/questions/30577375/have-numpy-argsort-return-an-array-of-2d-indices
    indices[c] = np.dstack(np.unravel_index(np.argsort(abs(image[:,:,c]).ravel()), (width, height)))
    #print(indices[c].shape)
    counter = 1
    while counter <= 10:
        print("{:8d}   {} {}".format(counter, indices[c][0][counter], image[indices[c][0][indices[c].shape[1]-counter][0], indices[c][0][indices[c].shape[1]-counter][1], c]))
        counter += 1

