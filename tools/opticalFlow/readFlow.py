import flowiz as fz
import matplotlib.pyplot as plt
import cv2 as cv


OpticalFlowFile = "1_2.flo"

img = fz.convert_from_file(OpticalFlowFile)
cv.imshow('OPtical Flow loaded from ' + OpticalFlowFile, img)
cv.waitKey(0)

print("end")