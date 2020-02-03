# Developed by: David Béjar Cáceres 2020
# This script compares the real optical flow (ground thruth) against the farnerback optical flow

import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
import ssim_error_flow
from ssim_error_flow import error_ssim_compareReal_Fernerback as ssim_real_vs_farnerback


flowDatasetRoot = os.sep + os.path.join("home", "nvidia", "tensorflowOpticalFlow", "MPI-Sintel", "training")

alley1Images = os.path.join(flowDatasetRoot, "final", "alley_1") + os.sep
alley2Images = os.path.join(flowDatasetRoot, "final", "alley_2") + os.sep
shaman1Images = os.path.join(flowDatasetRoot, "final", "shaman_1") + os.sep

alley1Flow = os.path.join(flowDatasetRoot, "flow", "alley_1") + os.sep
alley2Flow = os.path.join(flowDatasetRoot, "flow", "alley_2") + os.sep
shaman1Flow= os.path.join(flowDatasetRoot, "flow", "shaman_1") + os.sep


def main():
    print(alley1Images)
    print(alley2Images)
    print(shaman1Images)

    print(" -------------------- ")
    print(alley1Flow)
    print(alley2Flow)
    print(shaman1Flow)

    error = ssim_real_vs_farnerback(f'{alley1Images}frame_0001.png', f'{alley1Images}frame_0002.png', f'{alley1Flow}frame_0002.flo',)
    print("ERROR from bench: " + str(error))



    # for pair_alley1 in range(1, 50):
    #     image_path1 = f'{alley1Images}frame_00{pair_alley1:02d}.png'
    #     image_path2 = f'{alley1Images}frame_00{pair_alley1+1:02d}.png'

    return 0;

if __name__ == "__main__":
    main();
    pass