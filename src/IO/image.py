import cv2
import numpy as np
import os
import tempfile
tempDir = tempfile.gettempdir()

class InputFileException(Exception):
    pass

def read(prefix = tempDir, index = "000"):
    '''Read a 3-components image from disk. Each component stores
       integers between [0, 65535].

    Parameters
    ----------

        prefix : str.

            Path to the image in the file system, without extension.

    Returns
    -------

        (disk): [:,:,:].

            A color image, where each component is in the range
            [-32768, 32767].

    '''
    fn = prefix + index + ".png"
    image = cv2.imread(fn, -1)
    if image is None:
        raise InputFileException('IO::image:read: {} not found'.format(fn))
    else:
        if __debug__:
            print("IO::image:read: read {}".format(fn))
    buf = image.astype(np.float32)
    buf -= 32768.0
    return buf.astype(np.int16)

def write(image, prefix = tempDir, index = "000"):
    '''Write a 3-components image to disk. Each component stores integers
       between [0, 65536].

    Parameters
    ----------

        image : [:,:,:].

            The color image to write, where each component is in the range [-32768, 32768].

        prefix : str.

            Path to the image in the file system, without extension.

    Returns
    -------

        (disk) : [:,:,:].

            A color image.
    '''

    image = image.astype(np.float32)
    image += 32768.0
    image = image.astype(np.uint16)
    fn = prefix + index + ".png"
    cv2.imwrite(fn, image)
    if __debug__:
        print("IO::image:write: written {}".format(fn))

if __name__ == "__main__":

    img = read(tempDir, "000")
    write(img, tempDir, "000")
    print("IO::image:__main__: generated /tmp/000.png")
