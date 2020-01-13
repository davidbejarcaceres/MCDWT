#!/usr/bin/env python

'''
CUDA-accelerated Computer Vision functions
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import sys
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)

image1Path = os.path.join(thisPath, 'basketball1.png')
image2Path = os.path.join(thisPath, 'parque.jpg')


def main():
    setUp()
    test_cuda_upload_download()
    test_cudaarithm_arithmetic()
    test_cudaarithm_logical()
    test_cudaarithm_arithmetic()
    test_cudabgsegm_existence()
    test_cudacodec()
    test_cudacodec_writer_existence()
    test_cudafilters_laplacian()
    test_cudaimgproc()
    test_cudaimgproc_cvtColor()


def setUp():
    if not cv.cuda.getCudaEnabledDeviceCount():
        print("No CUDA-capable device is detected")
        exit()


def test_cuda_upload_download():
    npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
    cuMat = cv.cuda_GpuMat()
    cuMat.upload(npMat)

    np.allclose(cuMat.download(), npMat)


def test_cudaarithm_arithmetic():
    npMat1 = np.random.random((128, 128, 3)) - 0.5
    npMat2 = np.random.random((128, 128, 3)) - 0.5

    cuMat1 = cv.cuda_GpuMat()
    cuMat2 = cv.cuda_GpuMat()
    cuMat1.upload(npMat1)
    cuMat2.upload(npMat2)
    cuMatDst = cv.cuda_GpuMat(cuMat1.size(), cuMat1.type())

    np.allclose(cv.cuda.add(cuMat1, cuMat2).download(),
                                cv.add(npMat1, npMat2))

    cv.cuda.add(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(), cv.add(npMat1, npMat2))

    np.allclose(cv.cuda.subtract(cuMat1, cuMat2).download(),
                                cv.subtract(npMat1, npMat2))

    cv.cuda.subtract(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(),
                                cv.subtract(npMat1, npMat2))

    np.allclose(cv.cuda.multiply(cuMat1, cuMat2).download(),
                                cv.multiply(npMat1, npMat2))

    cv.cuda.multiply(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(),
                                cv.multiply(npMat1, npMat2))

    np.allclose(cv.cuda.divide(cuMat1, cuMat2).download(),
                                cv.divide(npMat1, npMat2))

    cv.cuda.divide(cuMat1, cuMat2, cuMatDst)
    np.allclose(
        cuMatDst.download(), cv.divide(npMat1, npMat2))

    np.allclose(cv.cuda.absdiff(cuMat1, cuMat2).download(),
                                cv.absdiff(npMat1, npMat2))

    cv.cuda.absdiff(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(),
                                cv.absdiff(npMat1, npMat2))

    np.allclose(cv.cuda.compare(cuMat1, cuMat2, cv.CMP_GE).download(),
                                cv.compare(npMat1, npMat2, cv.CMP_GE))

    cuMatDst1 = cv.cuda_GpuMat(cuMat1.size(), cv.CV_8UC3)
    cv.cuda.compare(cuMat1, cuMat2, cv.CMP_GE, cuMatDst1)
    np.allclose(cuMatDst1.download(),
                                cv.compare(npMat1, npMat2, cv.CMP_GE))

    np.allclose(cv.cuda.abs(cuMat1).download(),
                                np.abs(npMat1))

    cv.cuda.abs(cuMat1, cuMatDst)
    np.allclose(cuMatDst.download(), np.abs(npMat1))

    np.allclose(cv.cuda.sqrt(cv.cuda.sqr(cuMat1)).download(),
                                cv.cuda.abs(cuMat1).download())

    cv.cuda.sqr(cuMat1, cuMatDst)
    cv.cuda.sqrt(cuMatDst, cuMatDst)
    np.allclose(cuMatDst.download(),
                                cv.cuda.abs(cuMat1).download())

    np.allclose(cv.cuda.log(cv.cuda.exp(cuMat1)).download(),
                                npMat1)

    cv.cuda.exp(cuMat1, cuMatDst)
    cv.cuda.log(cuMatDst, cuMatDst)
    np.allclose(cuMatDst.download(), npMat1)

    np.allclose(cv.cuda.pow(cuMat1, 2).download(),
                                cv.pow(npMat1, 2))

    cv.cuda.pow(cuMat1, 2, cuMatDst)
    np.allclose(cuMatDst.download(), cv.pow(npMat1, 2))


def test_cudaarithm_logical():
    npMat1 = (np.random.random((128, 128)) * 255).astype(np.uint8)
    npMat2 = (np.random.random((128, 128)) * 255).astype(np.uint8)

    cuMat1 = cv.cuda_GpuMat()
    cuMat2 = cv.cuda_GpuMat()
    cuMat1.upload(npMat1)
    cuMat2.upload(npMat2)
    cuMatDst = cv.cuda_GpuMat(cuMat1.size(), cuMat1.type())

    np.allclose(cv.cuda.bitwise_or(cuMat1, cuMat2).download(),
                                cv.bitwise_or(npMat1, npMat2))

    cv.cuda.bitwise_or(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(),
                                cv.bitwise_or(npMat1, npMat2))

    np.allclose(cv.cuda.bitwise_and(cuMat1, cuMat2).download(),
                                cv.bitwise_and(npMat1, npMat2))

    cv.cuda.bitwise_and(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(),
                                cv.bitwise_and(npMat1, npMat2))

    np.allclose(cv.cuda.bitwise_xor(cuMat1, cuMat2).download(),
                                cv.bitwise_xor(npMat1, npMat2))

    cv.cuda.bitwise_xor(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(),
                                cv.bitwise_xor(npMat1, npMat2))

    np.allclose(cv.cuda.bitwise_not(cuMat1).download(),
                                cv.bitwise_not(npMat1))

    cv.cuda.bitwise_not(cuMat1, cuMatDst)
    np.allclose(cuMatDst.download(), cv.bitwise_not(npMat1))

    np.allclose(cv.cuda.min(cuMat1, cuMat2).download(),
                                cv.min(npMat1, npMat2))

    cv.cuda.min(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(), cv.min(npMat1, npMat2))

    np.allclose(cv.cuda.max(cuMat1, cuMat2).download(),
                                cv.max(npMat1, npMat2))

    cv.cuda.max(cuMat1, cuMat2, cuMatDst)
    np.allclose(cuMatDst.download(), cv.max(npMat1, npMat2))


def test_cudaarithm_arithmetic_420():
    npMat1 = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
    cuMat1 = cv.cuda_GpuMat(npMat1)
    cuMatDst = cv.cuda_GpuMat(cuMat1.size(), cuMat1.type())
    cuMatB = cv.cuda_GpuMat(cuMat1.size(), cv.CV_8UC1)
    cuMatG = cv.cuda_GpuMat(cuMat1.size(), cv.CV_8UC1)
    cuMatR = cv.cuda_GpuMat(cuMat1.size(), cv.CV_8UC1)

    np.allclose(cv.cuda.merge(cv.cuda.split(cuMat1)), npMat1)

    cv.cuda.split(cuMat1, [cuMatB, cuMatG, cuMatR])
    cv.cuda.merge([cuMatB, cuMatG, cuMatR], cuMatDst)
    np.allclose(cuMatDst.download(), npMat1)


def test_cudabgsegm_existence():
    # Test at least the existence of wrapped functions for now

    _bgsub = cv.cuda.createBackgroundSubtractorMOG()
    _bgsub = cv.cuda.createBackgroundSubtractorMOG2()

    # It is sufficient that no exceptions have been there
    


def test_cudacodec():
    # Test the functionality but not the results of the video reader

    vid_path = 'vtest.avi'
    try:
        reader = cv.cudacodec.createVideoReader(vid_path)
        ret, gpu_mat = reader.nextFrame()

        # TODO: print(cv.utils.dumpInputArray(gpu_mat)) # - no support for GpuMat

        # not checking output, therefore sepearate tests for different signatures is unnecessary
        ret, _gpu_mat2 = reader.nextFrame(gpu_mat)
        #TODO: gpu_mat == gpu_mat2)

    except cv.error as e:
        notSupported = (e.code == cv.Error.StsNotImplemented or e.code ==
                        cv.Error.StsUnsupportedFormat or e.code == cv.Error.GPU_API_CALL_ERROR)
        if e.code == cv.Error.StsNotImplemented:
            print("NVCUVID is not installed")
        elif e.code == cv.Error.StsUnsupportedFormat:
            print("GPU hardware video decoder missing or video format not supported")
        elif e.code == cv.Error.GPU_API_CALL_ERRROR:
            print("GPU hardware video decoder is missing")
        else:
            print(e.err)


def test_cudacodec_writer_existence():
    # Test at least the existence of wrapped functions for now

    try:
        _writer = cv.cudacodec.createVideoWriter("tmp", (128, 128), 30)
    except cv.error as e:
        #(e.code, cv.Error.StsNotImplemented)
        print("NVCUVENC is not installed")

    # It is sufficient that no exceptions have been there
    


def test_cudafeatures2d():
    npMat1 = self.get_sample("samples/data/right01.jpg")
    npMat2 = self.get_sample("samples/data/right02.jpg")

    cuMat1 = cv.cuda_GpuMat()
    cuMat2 = cv.cuda_GpuMat()
    cuMat1.upload(npMat1)
    cuMat2.upload(npMat2)

    cuMat1 = cv.cuda.cvtColor(cuMat1, cv.COLOR_RGB2GRAY)
    cuMat2 = cv.cuda.cvtColor(cuMat2, cv.COLOR_RGB2GRAY)

    fast = cv.cuda_FastFeatureDetector.create()
    _kps = fast.detectAsync(cuMat1)

    orb = cv.cuda_ORB.create()
    _kps1, descs1 = orb.detectAndComputeAsync(cuMat1, None)
    _kps2, descs2 = orb.detectAndComputeAsync(cuMat2, None)

    bf = cv.cuda_DescriptorMatcher.createBFMatcher(cv.NORM_HAMMING)
    matches = bf.match(descs1, descs2)
    self.assertGreater(len(matches), 0)
    matches = bf.knnMatch(descs1, descs2, 2)
    self.assertGreater(len(matches), 0)
    matches = bf.radiusMatch(descs1, descs2, 0.1)
    self.assertGreater(len(matches), 0)

    # It is sufficient that no exceptions have been there
    


def test_cudafilters_existence():
    # Test at least the existence of wrapped functions for now

    _filter = cv.cuda.createBoxFilter(cv.CV_8UC1, -1, (3, 3))
    _filter = cv.cuda.createLinearFilter(cv.CV_8UC4, -1, np.eye(3))
    _filter = cv.cuda.createLaplacianFilter(cv.CV_16UC1, -1, ksize=3)
    _filter = cv.cuda.createSeparableLinearFilter(
        cv.CV_8UC1, -1, np.eye(3), np.eye(3))
    _filter = cv.cuda.createDerivFilter(cv.CV_8UC1, -1, 1, 1, 3)
    _filter = cv.cuda.createSobelFilter(cv.CV_8UC1, -1, 1, 1)
    _filter = cv.cuda.createScharrFilter(cv.CV_8UC1, -1, 1, 0)
    _filter = cv.cuda.createGaussianFilter(cv.CV_8UC1, -1, (3, 3), 16)
    _filter = cv.cuda.createMorphologyFilter(
        cv.MORPH_DILATE, cv.CV_32FC1, np.eye(3))
    _filter = cv.cuda.createBoxMaxFilter(cv.CV_8UC1, (3, 3))
    _filter = cv.cuda.createBoxMinFilter(cv.CV_8UC1, (3, 3))
    _filter = cv.cuda.createRowSumFilter(cv.CV_8UC1, cv.CV_32FC1, 3)
    _filter = cv.cuda.createColumnSumFilter(cv.CV_8UC1, cv.CV_32FC1, 3)
    _filter = cv.cuda.createMedianFilter(cv.CV_8UC1, 3)

    # It is sufficient that no exceptions have been there
    


def test_cudafilters_laplacian():
    npMat = (np.random.random((128, 128)) * 255).astype(np.uint16)
    cuMat = cv.cuda_GpuMat()
    cuMat.upload(npMat)

    np.allclose(cv.cuda.createLaplacianFilter(cv.CV_16UC1, -1, ksize=3).apply(cuMat).download(),
                                cv.Laplacian(npMat, cv.CV_16UC1, ksize=3))


def test_cudaimgproc():
    npC1 = (np.random.random((128, 128)) * 255).astype(np.uint8)
    npC3 = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
    npC4 = (np.random.random((128, 128, 4)) * 255).astype(np.uint8)
    cuC1 = cv.cuda_GpuMat()
    cuC3 = cv.cuda_GpuMat()
    cuC4 = cv.cuda_GpuMat()
    cuC1.upload(npC1)
    cuC3.upload(npC3)
    cuC4.upload(npC4)

    imagenOriginal  = cv.imread(image1Path, cv.COLOR_BGR2RGB)
    edges = cv.Canny(imagenOriginal, 100, 200)

    # cv.imshow('Original', imagenOriginal)
    # cv.imshow('Edges', edges)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    cv.cuda.cvtColor(cuC3, cv.COLOR_RGB2HSV)
    cv.cuda.demosaicing(cuC1, cv.cuda.COLOR_BayerGR2BGR_MHT)
    cv.cuda.gammaCorrection(cuC3)
    cv.cuda.alphaComp(cuC4, cuC4, cv.cuda.ALPHA_XOR)
    cv.cuda.calcHist(cuC1)
    cv.cuda.equalizeHist(cuC1)
    cv.cuda.evenLevels(3, 0, 255)
    cv.cuda.meanShiftFiltering(cuC4, 10, 5)
    cv.cuda.meanShiftProc(cuC4, 10, 5)
    cv.cuda.bilateralFilter(cuC3, 3, 16, 3)
    cv.cuda.blendLinear



    imagenOriginal = cv.imread(image2Path)

    scale_percent = 50 # percent of original size
    width = int(imagenOriginal.shape[1] * scale_percent / 100)
    height = int(imagenOriginal.shape[0] * scale_percent / 100)
    size = (width, height)

    imagenOriginal = cv.resize(imagenOriginal, size)

    # cv.imshow('Original resized image by half', imagenOriginal)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    cuImagenOriginal = cv.cuda_GpuMat(imagenOriginal)
    
    cuEdges = cv.cuda_GpuMat(cv.imread(image2Path))

    # cv.imshow('Original Host>gpu>host', cuImagenOriginal.download())
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    cuImagenOriginal = cv.cuda.cvtColor(cuImagenOriginal, cv.COLOR_BGR2GRAY)

    # cv.imshow('Original Host>gpu>host', cuImagenOriginal.download())
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    detectorImagen = cv.cuda.createCannyEdgeDetector(200, 100)
    cuEdges = detectorImagen.detect(cuImagenOriginal)



    cv.cuda.meanShiftSegmentation(cuC4, 10, 5, 5).download()

    clahe = cv.cuda.createCLAHE()
    clahe.apply(cuC1, cv.cuda_Stream.Null())

    histLevels = cv.cuda.histEven(cuC3, 20, 0, 255)
    cv.cuda.histRange(cuC1, histLevels)

    detector = cv.cuda.createCannyEdgeDetector(0, 100)
    detector.detect(cuC1)


    cv.imshow('Original', cuImagenOriginal.download())
    cv.imshow('Edges', cuEdges.download())
    cv.waitKey(0)
    cv.destroyAllWindows()


    detector = cv.cuda.createHoughLinesDetector(3, np.pi / 180, 20)
    detector.detect(cuC1)

    detector = cv.cuda.createHoughSegmentDetector(3, np.pi / 180, 20, 5)
    detector.detect(cuC1)

    detector = cv.cuda.createHoughCirclesDetector(3, 20, 10, 10, 20, 100)
    detector.detect(cuC1)

    detector = cv.cuda.createGeneralizedHoughBallard()
    # BUG: detect accept only Mat!
    # Even if generate_gpumat_decls is set to True, it only wraps overload CUDA functions.
    # The problem is that Mat and GpuMat are not fully compatible to enable system-wide overloading
    #detector.detect(cuC1, cuC1, cuC1)

    detector = cv.cuda.createGeneralizedHoughGuil()
    # BUG: same as above..
    #detector.detect(cuC1, cuC1, cuC1)

    detector = cv.cuda.createHarrisCorner(cv.CV_8UC1, 15, 5, 1)
    detector.compute(cuC1)

    detector = cv.cuda.createMinEigenValCorner(cv.CV_8UC1, 15, 5, 1)
    detector.compute(cuC1)

    detector = cv.cuda.createGoodFeaturesToTrackDetector(cv.CV_8UC1)
    detector.detect(cuC1)

    matcher = cv.cuda.createTemplateMatching(cv.CV_8UC1, cv.TM_CCOEFF_NORMED)
    matcher.match(cuC3, cuC3)

    # It is sufficient that no exceptions have been there
    


def test_cudaimgproc_cvtColor():
    npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
    cuMat = cv.cuda_GpuMat()
    cuMat.upload(npMat)

    np.allclose(cv.cuda.cvtColor(cuMat, cv.COLOR_BGR2HSV).download(),
                                cv.cvtColor(npMat, cv.COLOR_BGR2HSV))



if __name__ == '__main__':
    main()
    pass
