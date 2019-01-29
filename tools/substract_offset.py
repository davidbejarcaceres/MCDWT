#!/usr/bin/env python

import cv2
import numpy as np
import sys
import argparse

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

parser = argparse.ArgumentParser(description = "Substracts (and clips to [0,255]) an offset to an image\n\n"
                                 "Example:\n\n"
                                 "  substract_offset -i ../sequences/stockholm/000 -o /tmp/000 -f 32640\n",
                                 formatter_class=CustomFormatter)

parser.add_argument("-i", "--input",
                    help="Input image", default="../sequences/stockholm/000")

parser.add_argument("-o", "--output",
                    help="Input image", default="/tmp/000")

parser.add_argument("-f", "--offset", type=int,
                    help="Offset", default=32768-128)

args = parser.parse_args()

input = args.input
output = args.output
offset = args.offset

image = cv2.imread(input, -1)

if __debug__:
    print("Max value at input: {}".format(np.amax(image)))
    print("Min value at input: {}".format(np.amin(image)))

image = np.clip(image, offset, offset+255)
image -= offset

if __debug__:
    print("Substracting {}".format(32768-128))

if __debug__:
    print("Max value at output: {}".format(np.amax(image)))
    print("Min value at output: {}".format(np.amin(image)))

cv2.imwrite(output, image.astype(np.uint8))
