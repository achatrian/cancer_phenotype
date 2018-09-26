import glob
import os
import cv2
import random
import numbers
import re

import numpy as np
from scipy.stats import mode
import torch
from torch.utils.data import Dataset
import imgaug as ia
from imgaug import augmenters as iaa


class CK5Dataset(Dataset):

    def __init__(self, dir_, mode, augment=False):
        self.mode = mode

        self.file_list = []
        self.label = []
        self.augment = augment

        thedir = os.path.join(dir_, mode)
        folders = [ name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name)) ]

        #Read data paths (images containing full glands)
        self.gt_files = glob.glob(os.path.join(dir, mode, '**','gland_gt_[0-9].png'), recursive=True)
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**', 'gland_gt_[0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**', 'gland_gt_[0-9][0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**', 'gland_gt_[0-9][0-9][0-9][0-9].png'), recursive=True))
        assert(self.gt_files); "Cannot be empty"
        self.img_files = [re.sub('gt', 'img', gtfile) for gtfile in self.gt_files]
        assert (self.img_files)

        self.dir = dir_

        self.randomcrop = RandomCrop(512)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.FrequencyNoiseAlpha(
                                       exponent=(-4, 0),
                                       first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                       second=iaa.ContrastNormalization((0.5, 2.0))
                                   )
                               ]),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __len__(self):
        return len(self.file_list)

    def adjust_gamma(self, img, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        # invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(img, table)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        gt_name = self.gt_files[idx]

        bgr_img = cv2.imread(img_name, -1)
        b, g, r = cv2.split(bgr_img)  # get b,g,r

        img = cv2.merge([r, g, b])  # switch it to rgb
        gt = cv2.imread(gt_name, -1)

        # scale img
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        # im aug
        img = self.randomcrop(img)
        if self.augment:
            img = self.seq.augment_image(img)

        # scale between 0 and 1 and swap the dimension
        img = img.transpose(2, 0, 1)/255.0

        # normalised img between -1 and 1
        img = [np.expand_dims((img - 0.5)/0.5, axis=0) for img in img]
        img = np.concatenate(img, axis=0)

        # get label from gt
        label = mode(gt[np.logical_and(gt > 0, gt != 4)], axis=None)[0]
        label = label if label.size > 0 else 0  # if no glands, give 0 label

        # convert to torch tensor
        img = torch.from_numpy(img).type(torch.float)
        label = torch.from_numpy(label).type(torch.long)

        return img, label


class RandomCrop(object):
    """Crops the given np.ndarray at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if img.ndim == 3:
            img = img[y1:y1+th, x1:x1+tw, :]
        else:
            img = img[y1:y1 + th, x1:x1 + tw]
        return img