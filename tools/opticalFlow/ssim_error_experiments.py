# Developed by: David Béjar Cáceres 2020
# This script compares the real optical flow (ground thruth) against the farnerback optical flow

import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
import ssim_error_flow
from ssim_error_flow import error_ssim_compareReal_Fernerback as ssim_real_vs_farnerback


cDrive = 'c:' + os.sep + "Users"
thisPathWindows = os.path.join(cDrive, "Public", "opticalFlowTensor", "MPI-Sintel", "training")
flowDatasetRootLinux = os.sep + os.path.join("home", "nvidia", "tensorflowOpticalFlow", "MPI-Sintel", "training")

flowDatasetRoot = flowDatasetRootLinux if sys.platform == "linux" else thisPathWindows

alley1Images = os.path.join(flowDatasetRoot, "final", "alley_1") + os.sep
alley2Images = os.path.join(flowDatasetRoot, "final", "alley_2") + os.sep
shaman1Images = os.path.join(flowDatasetRoot, "final", "shaman_2") + os.sep

alley1Flow = os.path.join(flowDatasetRoot, "flow", "alley_1") + os.sep
alley2Flow = os.path.join(flowDatasetRoot, "flow", "alley_2") + os.sep
shaman1Flow= os.path.join(flowDatasetRoot, "flow", "shaman_2") + os.sep

secuencias: str = [
    "alley_1.txt",
    "alley_2.txt", # moving a lot
    "shaman_2.txt",
    "cave_2.txt", # moving a lot++
    "market_5.txt", # moving a lot
    "sleeping_1.txt",
    "temple_3.txt" # moving a lot

]

def main():
    print(alley1Images)
    print(alley2Images)
    print(shaman1Images)

    print(" -------------------- ")
    print(alley1Flow)
    print(alley2Flow)
    print(shaman1Flow)

    # error = ssim_real_vs_farnerback(f'{shaman1Images}frame_0005.png', f'{shaman1Images}frame_0006.png', f'{shaman1Flow}frame_0005.flo', show = True)
    # print("ERROR from bench: " + str(error))

    for secuencia in range(0, len(secuencias)):
        resultados = open(f"ssim_error_{secuencias[secuencia]}", mode="w")
        for n_frame in range(1, 50):
            image_path1 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.png')
            image_path2 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame+1:02d}.png')
            realFlowPath = os.path.join(flowDatasetRoot, "flow", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.flo')

            error = ssim_real_vs_farnerback(image_path1, image_path2, realFlowPath, show = False)
            print("SSIM from bench: " + str(error))
            resultados.write(  str(error) + "\n")

        resultados.close()


    return 0;

def compareFlows_GUI():
    for secuencia in range(0, len(secuencias)):
        for n_frame in range(10, 11):
            image_path1 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.png')
            image_path2 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame+1:02d}.png')
            realFlowPath = os.path.join(flowDatasetRoot, "flow", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.flo')

            error = ssim_real_vs_farnerback(image_path1, image_path2, realFlowPath, show = True)
            nomSecuencia = secuencias[secuencia].replace(".txt", "")
            print("SSIM from sequence  " + nomSecuencia + ":  " + str(error))


if __name__ == "__main__":
    main();
    pass