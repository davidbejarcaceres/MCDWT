# Developed by: David Bejar Caceres 2020
# Using Scikit implementation: https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import cv2 as cv
import sys
import os
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)

N_threads = 6
iterations = 100
cv.setNumThreads(N_threads)

image1Path = os.path.join(thisPath, 'basketball1.png')
image2Path = os.path.join(thisPath, 'basketball2.png')


def main():
    frame1 = cv.imread(image1Path, cv.IMREAD_GRAYSCALE)
    frame2 = cv.imread(image2Path, cv.IMREAD_GRAYSCALE)
    #ssim_scikit(frame1, frame2)
    ssim_scikit_GUI(frame1, frame2)

    print("OK")


def ssim_scikit(frame1, frame2):
    img = frame1
    rows, cols = img.shape

    img_noise = frame2
    # mse_noise = mse(img, img_noise)
    ssim_noise = ssim(
        img, img_noise, data_range=img_noise.max() - img_noise.min())

    return ssim_noise


def ssim_scikit_GUI(frame1, frame2):
    img = img_as_float(frame1)
    rows, cols = img.shape

    # noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
    # noise[np.random.random(size=noise.shape) > 0.5] *= -1

    img_noise = img_as_float(frame2)
    img_const = img

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    mse_none = mse(img, img)
    ssim_none = ssim(img, img, data_range=img.max() - img.min())

    mse_noise = mse(img, img_noise)
    ssim_noise = ssim(img, img_noise,
                      data_range=img_noise.max() - img_noise.min())

    mse_const = mse(img, img_const)
    ssim_const = ssim(img, img_const,
                      data_range=img_const.max() - img_const.min())

    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[0].set_xlabel(label.format(mse_none, ssim_none))
    ax[0].set_title('Original image')

    ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[1].set_xlabel(label.format(mse_noise, ssim_noise))
    ax[1].set_title('Image with noise')

    ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax[2].set_xlabel(label.format(mse_const, ssim_const))
    ax[2].set_title('Image plus constant')

    plt.tight_layout()
    plt.show()

    return ssim_noise


def mse(x, y):
    return np.linalg.norm(x - y)


if __name__ == "__main__":
    main()
    pass
