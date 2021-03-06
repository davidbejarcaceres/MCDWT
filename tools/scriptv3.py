#!/usr/bin/env python3

# Note: swap the above line with the following two ones to switch
# between the standard and the optimized running mode.

#!/bin/sh
''''exec python3 -O -- "$0" ${1+"$@"} # '''

"""scriptv3.py: video DMCWT multi-level transform and reconstruction utility with Windows an Linux support."""

# course: Multimedia Systems
# Issue: 1
# date: 2019
# username: dbc770
# name: David Béjar Cáceres
# description: video DMCWT multi-level direct transform and reconstruction utility.

import numpy as np
import pywt
import math
import sys
import os
import subprocess
import argparse
import cv2
import numpy
import tempfile
projectPathOS = sys.path[0].replace("/tools", "") if sys.platform == "linux" else sys.path[0].replace("\\tools", "")
sys.path.append(projectPathOS)
sys.path.insert(0, "..")
pythonversion3OS = "python3.6" if sys.platform == "linux" else "python"
tempPath = tempfile.gettempdir()
import src.DWT
import src.IO
import src.MC
import requests
import shutil
import add_offset_module as add_offset
import substract_offset_module as substract_offset


def main():
    # Creates the command line arguments
    parser = argparse.ArgumentParser("Script for Issue Week 2")
    parser.add_argument("-vpath", help="Path to the local video, with .mp4 extension")
    parser.add_argument("-vurl", help="URL to the video to download")
    parser.add_argument("-level", help="Number of spatial resolutions (levels in the Laplacian Pyramid)" ,default=1)
    parser.add_argument("-frames", help="Number of frames to extract from video to transform", default=5)
    parser.add_argument("-T", help="Number of levels of the MCDWT (temporal scales)", default=2)
    parser.add_argument("-vname", help="Name of the folder and video to export in tmp folder")
    parser.add_argument("-b", "--backward", action='store_true', help="Performs backward transform")


    # Parses all the arguments
    args = parser.parse_args()

    # URL to the video
    videoURL = args.vurl if(args.vurl != None) else "http://www.hpca.ual.es/~vruiz/videos/un_heliostato.mp4"

    # Path to the video in local
    if args.vpath != None:
        videoPath = args.vpath
        localVideo = True
    else:
        localVideo = False

    # Name of the video
    videoName = args.vname if(args.vname != None) else "video_transformed"

    # Number of frames to be extracted
    nFrames = int(args.frames) if(args.frames != None) else 5

    # Number of T
    t_scale = int(args.T) if(args.T != None) else 2

    # Number of levels for the Laplacian Pyramid
    nLevel = int(args.level) if(args.level != None) else 1

    if not args.backward:
        # Creates directories for the generated files
        os.makedirs(os.path.join(tempPath, videoName), 755, 1)
        os.makedirs(os.path.join(tempPath, videoName, "extracted"), 755, 1)
        os.makedirs(os.path.join(tempPath, videoName, "16bit"), 755, 1)
        os.makedirs(os.path.join(tempPath, videoName, "reconstructed"), 755, 1)

        # Working with videos from the web
        if localVideo != True:
            # Downloads the video
            print("Atempting to download video ...\n\n")
            #subprocess.run("wget {} -O /tmp/{}/{}.mp4 ".format(videoURL, videoName, videoName) , shell=True, check=True)
            # To save to an absolute path.
            r = requests.get("http://www.hpca.ual.es/~vruiz/videos/un_heliostato.mp4")  
            with open(os.path.join(tempPath, videoName, videoName+".mp4"), 'wb') as f:
                f.write(r.content)
            print("\n\nVideo Downloaded!\n\n")

        # Working with local video
        if localVideo:
            shutil.copyfile(videoPath, os.path.join(tempPath, videoName, videoName+".mp4")) 

        
        # Extracts the frames from video using OpenCV
        cap = cv2.VideoCapture(os.path.join(tempPath,videoName, videoName+".mp4"))
        extracted = 0
        while extracted < nFrames:
            success, frame = cap.read()
            salida = "{}{}{}_{:03d}".format(os.path.join(tempPath,videoName,"extracted"), os.sep, videoName , extracted+1)
            cv2.imwrite("{}.png".format(salida), frame)   # save frame as JPEG file
            success, frame = cap.read()
            extracted += 1

        # print("\n\nExtracting images ...\n\n")
        # subprocess.run("ffmpeg -i {} -vframes {} {}_%03d.png".format(os.path.join(tempPath,videoName, videoName+".mp4"),  nFrames, os.path.join(tempPath,videoName,"extracted", videoName)), shell=True, check=True)

        # Convert the images to 16 bit
        for image in range(int(nFrames)):
            inputImg = ("{}_{:03d}.png".format( os.path.join(tempPath, videoName, "extracted", videoName) ,image+1))
            outputImg = ("{}{:03d}.png".format( os.path.join(tempPath, videoName, "16bit") + os.path.sep ,image))
            add_offset.add_offset(inputImg, outputImg)

        # delete extensions from 16 bit images
        # for image in range(int(nFrames)):
        #     subprocess.run("mv /tmp/{}/16bit/{:03d}.png /tmp/{}/16bit/{:03d}".format(videoName, image, videoName, image), shell=True, check=True)

        print("\n Removed extensions from 16 bit images...\n")
        print("\nDone! ready to transform \n")

        ########## MDWT Transform #################
        # Motion 2D 1-levels forward DWT of the frames from the video:
        subprocess.run("{}  {}MDWT.py -p {}  -N {}".format(pythonversion3OS,os.path.join(projectPathOS, "src") + os.path.sep, os.path.join(tempPath, videoName, "16bit") + os.path.sep, nFrames), shell=True, check=True)
        print("\nFirst transform MDWT done!")

        ########## MCDWT Transform #################
        # Motion Compensated 1D 1-levels forward MCDWT:
        subprocess.run("{}  {} -p {}  -N {} -T {}".format(pythonversion3OS ,os.path.join(projectPathOS, "src") + os.path.sep + "MCDWT.py", os.path.join(tempPath, videoName, "16bit") + os.path.sep, nFrames, t_scale), shell=True, check=True)
        print("\nTransform MCDWT done!")

        newLevel = "" # Stores the last level of the MCDWT

        # Support for multi-level forwards MCDWT
        if nLevel > 1:
            print("\nWorking on multi-level MCDWT...")
            mlevelpath = os.path.join(tempPath, videoName, "MDWT", "MCDWT") + os.path.sep
            for i in range(nLevel-1):
                print(f"Nivel {i+2}")
                newLevel = (f"{mlevelpath}{os.path.sep}{str(i+2)}")
                os.makedirs(newLevel, 755, 1)
                # Works on the last MCDWT and transform to the new level
                subprocess.run("{}  -O {} -d {}{} -m {}{} -N {} -T {}".format(pythonversion3OS, os.path.join(projectPathOS, "src") + os.path.sep + "MCDWT.py", mlevelpath, i+1 + os.path.sep, mlevelpath, i+2 + os.path.sep, nFrames, t_scale), shell=True, check=True)


            print("Last level of MCDWT located in:  {}".format(newLevel))

    if args.backward:
        ######## Reconstruction #########
        # Support for multi-level backwards reconstruction MCDWT
        if nLevel > 1:
            print("\nReconstructing multi-level MCDWT...")
            mlevelMCDWT = os.path.join(tempPath, videoName, "MDWT", "MCDWT") + os.path.sep
            mlevelrecons = os.path.join(tempPath, videoName, "_reconMCDWT") + os.path.sep
            for i in reversed(range(nLevel-1)):
                os.makedirs("{}{}".format(mlevelrecons, i+1),755, 1) # creates next level MCDWT dir
                # Reconstructs form the last MCDWT level
                subprocess.run("{}  -O {} -b -m {}{}  -d {}{}  -N {} -T {}".format(pythonversion3OS, os.path.join(projectPathOS, "src") + os.path.sep + "MCDWT.py",     mlevelMCDWT, i+2 + os.path.sep,    mlevelrecons   , i+1 + os.path.sep    , nFrames-1, t_scale), shell=True, check=True)
                print("Reconstructed from MCDWT level {} done!".format(i+2))

            print("\nMulti-levels reconstructed...\n")
        else:
            # Reconstructs form the last MCDWT level
                print("\nReconstructed from level MCDWT")
        subprocess.run("{}  {} -p {}  -N {} -T {}".format(pythonversion3OS, os.path.join(projectPathOS, "src") + os.path.sep + "MCDWT.py", os.path.join(tempPath, videoName, "16bit") + os.path.sep, nFrames, t_scale), shell=True, check=True)

        # Motion 2D 1-levels backward DWT:
        subprocess.run("{}  {} -p {} -b  -N {}".format(pythonversion3OS ,os.path.join(projectPathOS, "src") + os.path.sep + "MCDWT.py", os.path.join(tempPath, videoName, "16bit") + os.path.sep, nFrames), shell=True, check=True)
        print("\nReconstructed from MDWT done!")
        print("\nCheck reconstructed sequence in: {}/_reconMDWTcon".format(tempPath))

        # Reconstruct 16bit original images back to normal
        for image in range(int(nFrames)):
            inputImg = ("{}{:03d}.png".format(os.path.join(tempPath, videoName, "16bit") + os.path.sep ,image))
            outputImg = ("{}{:03d}.png".format(os.path.join(tempPath, videoName, "reconstructed") + os.path.sep ,image))
            substract_offset.substract_offset(inputImg, outputImg)

    print("Check results on {} folder".format(os.path.join(tempPath, videoName)))
    print("\n\nScript Finished!")

if __name__ == "__main__":
    main()
