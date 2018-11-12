# @Author: Sharib Ali <shariba>
# @Date:   2018-05-11T14:01:34+01:00
# @Email:  sharib.ali@eng.ox.ac.uk
# @Project: BRC: VideoEndoscopy
# @Filename: singleImageQuality.py
# @Last modified by:   shariba
# @Last modified time: 2018-05-11T14:08:21+01:00
# @Copyright: 2018-2020, sharib ali
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def VOLA(image):
    # compute correlation (Santos'97)
    #I1(1:end-1,:) = image(2:end,:)
    debug = 0
    img = im2double(image)
    h, w = img.shape[:2]
    I1 = np.zeros((h, w), np.double)
    I2 = np.zeros((h, w), np.double)

    I1[0:-2,:] = img[2:,:]
    I2[0:-3,:] = img[3:,:]
    I = I1-I2
    image =image * I
    if debug:
        cv2.imshow('diff', image)
        cv2.waitKey(0)
    meanFM, stdFM = cv2.meanStdDev(image)
    return meanFM
