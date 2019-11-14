#!/usr/bin/env python

import cv2
import numpy as np
import sys
import argparse
import os
import tempfile

projectPathOS = os.getcwd()
sys.path.append(projectPathOS)
projectPathOS = projectPathOS.replace("/tools", "") if sys.platform == "linux" else projectPathOS.replace("\\tools", "")
sys.path.append(projectPathOS)


def substract_offset(input, output, offset=32768-128):
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