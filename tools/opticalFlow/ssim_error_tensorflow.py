import numpy as np 
import cv2 as cv
import sys
import os
import tensorflow as tf
from tensorflow.keras.backend import eval
import argparse
thisPath = sys.path[0]
filesPath = os.listdir(thisPath)

image1Path = os.path.join(thisPath, 'basketball1.png')
image2Path = os.path.join(thisPath, 'basketball2.png')

def main():
    parser = argparse.ArgumentParser(description = "Returns the ssim error of two png images using Tensorflow Library\n\n"
                                 "Example:\n\n"
                                 f"  ssim_error_scikit -i {image1Path} -j {image2Path} \n")

    parser.add_argument("-i", "--frame1",
                        help="Input image 1", default=image1Path) #"../sequences/stockholm/000"

    parser.add_argument("-j", "--frame2",
                        help="Input image 2", default=image2Path)
    args = parser.parse_args()


    frame1 = cv.imread( args.frame1 )
    frame2 = cv.imread( args.frame2 )
    if frame1 is None:
        print("ERROR: File not found:  " + args.frame1)
        exit()

    if frame2 is None:
        print("ERROR: File not found:  " + args.frame2)
        exit()

    img1 = tf.convert_to_tensor(frame1)
    img2 = tf.convert_to_tensor(frame2)

    error_ssim = get_ssim_tensorFlow(img1, img2)
    
    print("SSIM Error:")
    print(eval(error_ssim))

    pass

def get_ssim_tensorFlow(frame1: tf.dtypes.uint8, frame2: tf.dtypes.uint8, maxRange: int = 255):
    return tf.image.ssim(frame1, frame2, max_val = maxRange, filter_sigma=1.5)

def get_ssim_tensorFlow_float32(frame1: tf.dtypes.uint8, frame2: tf.dtypes.uint8, maxRange: int = 255):
    img1 = tf.convert_to_tensor(frame1) # 3 color channel opencv image
    img2 = tf.convert_to_tensor(frame2) # 3 color channel opencv image
    img1 = tf.image.convert_image_dtype(frame1, tf.float32)
    img2 = tf.image.convert_image_dtype(frame2, tf.float32)
    return tf.image.ssim(img1, img2, max_val = maxRange, filter_sigma=1.5)



def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    img1 = image_to_4d(img1.astype(np.float32))
    img2 = image_to_4d(img2.astype(np.float32))
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value



if __name__ == "__main__":
    main()
    pass