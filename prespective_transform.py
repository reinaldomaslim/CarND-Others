import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob 
%matplotlib inline
%matplotlib qt

#Read in calibration images
img = mpimg.imread('../calibration_images/calibration1.jpg')
plt.imshow(img)

plt.imshow(img)
#select four points of a plane on original image
plt.plot(850, 320, '.')
plt.plot(865, 450, '.')
plt.plot(533, 350, '.')
plt.plot(535, 210, '.')


def warp(img):
    img_size=(img.shape[1], image.shape[0])

    #Four source coordinates
    src=np.float32(
        [[850, 320],
        [865, 450],
        [533, 350],
        [535, 210]])

    #Four destination coordinates
    dst=np.float32([
        [870, 240],
        [870, 370],
        [520, 370],
        [520, 240]])

    N=cv2.getPerspectiveTransform(src, dst)

    Ninv=cv2.getPerspectiveTransform(dst, src)

    warped=cv2.warpPerspective(img, N, img_size, flags=cv2.INTER_LINEAR)


    return warped



