#!/usr/bin/env python

import cv2
import argparse
import tempfile
import sys
import os
projectPathOS = sys.path[0].replace(
    "/tools", "") if sys.platform == "linux" else sys.path[0].replace("\\tools", "")
sys.path.append(projectPathOS)
sys.path.insert(0, "..")
pythonversion3OS = "python3.6" if sys.platform == "linux" else "python"
tempPath = tempfile.gettempdir()


# Parse arguments
parser = argparse.ArgumentParser(
    description='This script converts a YCbCr image to RGB.')
parser.add_argument('-i', '--input', help='Input file name', required=True)
parser.add_argument('-o', '--output', help='Output file name', required=True)
args = parser.parse_args() + os.sep

# Read image
image_ycc = cv2.imread(args.input, -1)
if image_ycc is None:
    raise Exception('{} not found'.format(args.input))

# Convert to YCbCr
image_rgb = cv2.cvtColor(image_ycc, cv2.COLOR_YCR_CB2BGR)

# Write image
cv2.imwrite(args.output, image_rgb)


def ycc2rgb_cuda(input: str = os.path.join(projectPathOS, "sequences", "stockholm", "000.png"), output: str = (tempPath + os.sep + "imagergb.png")):
    image_ycc = cv2.imread(input, -1)
    if image_ycc is None:
        raise Exception('{} not found'.format(args.input))

    cuImage_ycc = cv2.cuda_GpuMat()  # Allocates memory on the GPU
    cuImage_ycc.upload(npMat1)  # Moves data from Host to GPU memory
    cuImage_ycc = cv2.cuda.cvtColor(
        cuImage_ycc, cv2.COLOR_RGB2GRAY)  # Conversion on the GPU
    image_rgb = cuImage_ycc.download()
    cv2.imwrite(output, image_rgb)
