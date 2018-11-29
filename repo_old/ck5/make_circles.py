import os
import sys
import argparse

import numpy as np
import scipy as sp

import imageio
from skimage.transform import warp
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_A', required=True)
    parser.add_argument('--fold_B', required=True)
    args, unparsed = parser.parse_known_args()



sift = cv2.xfeatures2d.SIFT_create()
def make_circles(args):
    gts = os.listdir(args.fold_A)

    for gt in gts:
        gt = imageio.imread(gt)
        kp = sift.detect(gt ,None)



