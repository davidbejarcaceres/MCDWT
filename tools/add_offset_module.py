import cv2
import numpy as np
import sys
import argparse
from subprocess import check_call
from subprocess import CalledProcessError

def add_offset(input, output, offset=32768-128):

    image = cv2.imread(input, -1).astype(np.uint16)

    if __debug__:
        print("Max value at input: {}".format(np.amax(image)))
        print("Min value at input: {}".format(np.amin(image)))

    if __debug__:
        print("Adding {}".format(offset))

    image += offset

    if __debug__:
        print("Max value at output: {}".format(np.amax(image)))
        print("Min value at output: {}".format(np.amin(image)))

    cv2.imwrite(output + ".png", image.astype(np.uint16))
    try:
        check_call("mv " + output + ".png " + output, shell=True)
    except CalledProcessError:
        print("Exception {}".format(traceback.format_exc()))

    pass

