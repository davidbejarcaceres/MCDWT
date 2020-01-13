# David Bejar Caceres

import cv2
import tempfile
import sys
import os
projectPathOS = sys.path[0].replace(
    "/tools", "") if sys.platform == "linux" else sys.path[0].replace("\\tools", "")
sys.path.append(projectPathOS)
sys.path.insert(0, "..")
pythonversion3OS = "python3.6" if sys.platform == "linux" else "python"
tempPath = tempfile.gettempdir()


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