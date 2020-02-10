# Developed by: David Béjar Cáceres 2020
# This script compares the real optical flow (ground thruth) against the farnerback optical flow

import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
import ssim_error_flow
from ssim_error_flow import error_ssim_compareReal_Fernerback as ssim_real_vs_farnerback
from ssim_error_flow import error_ssim_compareReal_Dual_TVL1 as ssim_real_vs_Dual_TVL1



cDrive = 'c:' + os.sep + "Users"
# thisPathWindows = os.path.join(cDrive, "Aulas_Biblioteca","Documents" , "David", "tensorflowOpticalFlow", "MPI-Sintel", "training")
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

secuenciasDualTV1: str = [
    "dualTV1_alley_1.txt",
    "dualTV1_alley_2.txt", # moving a lot
    "dualTV1_shaman_2.txt",
    "dualTV1_cave_2.txt", # moving a lot++
    "dualTV1_market_5.txt", # moving a lot
    "dualTV1_sleeping_1.txt",
    "dualTV1_temple_3.txt" # moving a lot

]

def main():
    print(alley1Images)
    print(alley2Images)
    print(shaman1Images)

    print(" -------------------- ")
    print(alley1Flow)
    print(alley2Flow)
    print(shaman1Flow)

    print(" -------------------- ")

    farnerback(nFrames=50)
    DualTV1(nFrames=50)

    # error = ssim_real_vs_farnerback(f'{shaman1Images}frame_0005.png', f'{shaman1Images}frame_0006.png', f'{shaman1Flow}frame_0005.flo', show = True)
    # print("ERROR from bench: " + str(error))


    return 0;


def farnerback(nFrames=5):
    if nFrames > 50:
        nFrames = 50
    print("Running experiments using Farnerback Method")
    for secuencia in range(0, len(secuencias)):
        resultados = open(f"ssim_error_{secuencias[secuencia]}", mode="w")
        resultadosMSE = open(f"mse_error_{secuencias[secuencia]}", mode="w")
        for n_frame in range(1, nFrames):
            image_path1 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.png')
            image_path2 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame+1:02d}.png')
            realFlowPath = os.path.join(flowDatasetRoot, "flow", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.flo')

            error = ssim_real_vs_farnerback(image_path1, image_path2, realFlowPath, show = True)
            print("SSIM error: " + str(error[0]))
            print("MSE error: " + str(error[1]))
            resultados.write(  str(error[0]) + "\n")
            resultadosMSE.write(  str(error[1]) + "\n")

        resultados.close()
    return 0

def DualTV1(nFrames=5):
    if nFrames > 50:
        nFrames = 50
    print("Running experiments using DualTVL1 method")
    for secuencia in range(0, len(secuencias)):
        resultados = open(f"ssim_error_{secuenciasDualTV1[secuencia]}", mode="w")
        resultadosMSE = open(f"mse_error_{secuenciasDualTV1[secuencia]}", mode="w")
        for n_frame in range(1, nFrames):
            image_path1 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.png')
            image_path2 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame+1:02d}.png')
            realFlowPath = os.path.join(flowDatasetRoot, "flow", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.flo')

            error = ssim_real_vs_Dual_TVL1(image_path1, image_path2, realFlowPath, show = False)
            print("SSIM error: " + str(error[0]))
            print("MSE error: " + str(error[1]))
            resultados.write(  str(error[0]) + "\n")
            resultadosMSE.write(  str(error[1]) + "\n")

        resultados.close()
    return 0


def compareFlows_GUI_Farneback(nFrames=5):
    if nFrames > 50:
        nFrames = 50
    for secuencia in range(0, len(secuencias)):
        for n_frame in range(1, nFrames):
            image_path1 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.png')
            image_path2 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame+1:02d}.png')
            realFlowPath = os.path.join(flowDatasetRoot, "flow", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.flo')

            error = ssim_real_vs_farnerback(image_path1, image_path2, realFlowPath, show = True)
            nomSecuencia = secuencias[secuencia].replace(".txt", "")
            print("SSIM from sequence  " + nomSecuencia + ":  " + str(error))


def compareFlows_GUI_Dual_TVL1(nFrames=5):
    if nFrames > 50:
        nFrames = 50
    for secuencia in range(0, len(secuencias)):
        for n_frame in range(1, nFrames):
            image_path1 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.png')
            image_path2 = os.path.join(flowDatasetRoot, "final", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame+1:02d}.png')
            realFlowPath = os.path.join(flowDatasetRoot, "flow", secuencias[secuencia].replace(".txt", ""), f'frame_00{n_frame:02d}.flo')

            error = ssim_real_vs_Dual_TVL1(image_path1, image_path2, realFlowPath, show = True)
            nomSecuencia = secuencias[secuencia].replace(".txt", "")
            print("SSIM from sequence  " + nomSecuencia + ":  " + str(error))


if __name__ == "__main__":
    main();
    pass