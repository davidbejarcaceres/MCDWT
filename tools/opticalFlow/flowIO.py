import os
import numpy as np
import cv2 as cv
import sys
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)
import OpticalFlowToColor

TAG_FLOAT = 202021.25
TAG_STRING = 'PIEH'

flowFile = os.path.join(thisPath, "1_2.flo")

def main():
    flow = readFlow(flowFile)
    flowColor = OpticalFlowToColor.flow_to_color(flow)

    cv.imshow('Optical Flow loaded from ' + flowFile, flowColor)
    cv.waitKey(0)


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)
    

def read(file: str) -> np.uint8:

	assert type(file) is str, "file is not str %r" % str(file)
	assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
	assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
	f = open(file,'rb')
	flo_number = np.fromfile(f, np.float32, count=1)[0]
	assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
	w = np.fromfile(f, np.int32, count=1)
	h = np.fromfile(f, np.int32, count=1)
	#if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
	data = np.fromfile(f, np.float32, count=2*w[0]*h[0])

	# data = np.fromfile(f, np.float32, count=2*w*h)
	# Reshape data into 3D array (columns, rows, bands)
	flow = np.resize(data, (int(h), int(w), 2))	
	f.close()

	return flow


def write(flow, filename):

	assert type(filename) is str, "file is not str %r" % str(filename)
	assert filename[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]

	height, width, nBands = flow.shape
	assert nBands == 2, "Number of bands = %r != 2" % nBands
	u = flow[: , : , 0]
	v = flow[: , : , 1]	
	assert u.shape == v.shape, "Invalid flow shape"
	height, width = u.shape

	f = open(filename,'wb')
	f.write(TAG_STRING)
	np.array(width).astype(np.int32).tofile(f)
	np.array(height).astype(np.int32).tofile(f)
	tmp = np.zeros((height, width*nBands))
	tmp[:,np.arange(width)*2] = u
	tmp[:,np.arange(width)*2 + 1] = v
	tmp.astype(np.float32).tofile(f)

	f.close()

if __name__ == "__main__":
    main()
    pass

