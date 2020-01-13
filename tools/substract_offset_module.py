#!/usr/bin/env python

import cv2
import numpy as np
import sys
import argparse
import os
import tempfile
import argparse
import tempfile
tempPath = tempfile.gettempdir()
projectPathOS = os.getcwd()
sys.path.append(projectPathOS)
projectPathOS = projectPathOS.replace("/tools", "") if sys.platform == "linux" else projectPathOS.replace("\\tools", "")
sys.path.append(projectPathOS)


def main():
    parser = argparse.ArgumentParser(description = "Substracts (and clips to [0,255]) an offset to an image\n\n"
                                 "Example:\n\n"
                                 "  substract_offset -i ../sequences/stockholm/000 -o /tmp/000 -f 32640\n")

    parser.add_argument("-i", "--input",
                        help="Input image", default=os.path.join(projectPathOS, "sequences", "stockholm", "000.png")) #"../sequences/stockholm/000"

    parser.add_argument("-o", "--output",
                        help="Input image", default=os.path.join(tempPath, "000.png"))

    parser.add_argument("-f", "--offset", type=int,
                        help="Offset", default=32768-128)

    args = parser.parse_args()

    input = args.input
    output = args.output
    offset = args.offset

    substract_offset(input, output)



def substract_offset(input, output, offset=32768-128):
    image = cv2.imread(input, -1)

    if image is None: 
        print("ERROR: File not found in: " + input)
        exit()

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

if __name__ == "__main__":
    main()
    pass