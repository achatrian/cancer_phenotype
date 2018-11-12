import glob
import os
import sys
import random
import numbers
import re
import warnings
from pathlib import Path
from itertools import product
import csv

import numpy as np
import cv2
from scipy.stats import mode
import torch
from torch.utils.data import Dataset
import imgaug as ia
from imgaug import augmenters as iaa

sys.path.append("../segment")
from segment.utils import is_pathname_valid


class CK5Dataset(Dataset):

    def __init__(self, dir_, mode, augment=False, tile_size=299):
        self.mode = mode

        self.label = []
        self.augment = augment
        self.tile_size = tile_size

        # Read data paths (images containing gland parts)
        self.gt_files = glob.glob(os.path.join(dir_, mode, '**','gland_gt_[0-9]_([0-9],[0-9]).png'), recursive=True)
        n = "[0-9]"
        for gl_idx, x, y in product(range(1, 4), range(1, 5), range(1, 5)):
            to_glob = os.path.join(dir_, mode, '**', 'gland_gt_' + n*gl_idx + '_(' + n*x + ',' + n*y + ').png')
            self.gt_files += glob.glob(to_glob, recursive=True)

        assert self.gt_files; r"Cannot be empty"
        self.img_files = [re.sub('gt', 'img', gtfile) for gtfile in self.gt_files]
        assert self.img_files

        self.dir = dir_

        self.randomcrop = RandomCrop(299)  # for inception patches are 299

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
        return len(self.gt_files)

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

        bg_mask = np.isclose(gt.squeeze(), 0)  # background mask
        stromal_mean = [np.mean(img[bg_mask, 0]), np.mean(img[bg_mask, 1]), np.mean(img[bg_mask, 1])]

        # im cropping / padding
        if img.shape[0] < self.tile_size or img.shape[1] < self.tile_size:
            left = (self.tile_size - img.shape[1]) // 2
            right = (self.tile_size - img.shape[1]) - left
            bottom = (self.tile_size - img.shape[0]) // 2
            top = (self.tile_size - img.shape[0]) - bottom

            left, right, bottom, top = max(left, 0), max(right, 0), max(bottom, 0), max(top, 0)

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=stromal_mean)
            gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

        if img.shape[0] >= self.tile_size or img.shape[1] >= self.tile_size:
            img = self.randomcrop(img)

        #img[gt == 0, 0] = stromal_mean[0]
        #img[gt == 0, 1] = stromal_mean[1]
        #img[gt == 0, 2] = stromal_mean[2]

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

        if label == 3:
            label = 1  # 3 classes need labels from 0 to 2 in loss computation

        # convert to torch tensor
        img = torch.from_numpy(img).type(torch.float)
        label = torch.tensor(label).type(torch.long).squeeze()

        return img, label


class AugDataset(CK5Dataset):

    def __init__(self, dir_, aug_dir, mode, augment=0, generated_only=True):
        super(AugDataset, self).__init__(dir_, mode, augment)
        if generated_only:
            self.gt_files = []
            self.img_files = []
        n = "[0-9]"
        names = ["_rec1_", "_rec2_", "_gen1_", "_gen2_"]
        no_aug_len = len(self.img_files)
        file_list = []
        for gl_idx, x, y, name in product(range(1, 7), range(1, 6), range(1, 6), names):
            to_glob = os.path.join(aug_dir, 'gland_img_' + n * gl_idx + '_(' + n * x + ',' + n * y + ')' + \
                                   name + 'fake_B.png')
            file_list += glob.glob(to_glob)
        self.img_files += file_list
        self.gt_files += [x.replace('fake_B', 'real_A') for x in file_list]
        assert (len(self.img_files) > no_aug_len)


class CK5AugmentStudyDataset(Dataset):

    def __init__(self, dir_, mode, augment=0, tile_size=299):
        self.mode = mode

        self.dir = dir_
        self.label = []
        self.augment = augment
        self.tile_size = tile_size

        # Read data paths (images containing full glands)
        self.gt_files = glob.glob(os.path.join(self.dir, mode, '**', 'gland_gt_[0-9].png'), recursive=True)
        self.gt_files.extend(glob.glob(os.path.join(self.dir, mode, '**', 'gland_gt_[0-9][0-9].png'), recursive=True))
        self.gt_files.extend(
            glob.glob(os.path.join(self.dir, mode, '**', 'gland_gt_[0-9][0-9][0-9].png'), recursive=True))
        self.gt_files.extend(
            glob.glob(os.path.join(self.dir, mode, '**', 'gland_gt_[0-9][0-9][0-9][0-9].png'), recursive=True))
        assert (self.gt_files);
        "Cannot be empty"
        self.img_files = [re.sub('gt', 'img', gtfile) for gtfile in self.gt_files]
        assert (self.img_files)



        self.randomcrop = RandomCrop(299)  # for inception patches are 299

        sometimes = lambda aug: iaa.Sometimes(0.7, aug)

        if augment == 1:
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
                    ))]
            )

        elif augment == 2:
            self.seq = self.seq = iaa.Sequential(
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
                                   ]))
                               ]
                               )
                ]
            )

        elif augment == 3:
            alpha = 0.5
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
                    iaa.SomeOf((0, 4),
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
                                   iaa.Sharpen(alpha=(0, alpha), lightness=(0.75, 1.5)),  # sharpen images
                                   iaa.Emboss(alpha=(0, alpha), strength=(0, 2.0)),  # emboss images
                                   # search either for all edges or for directed edges,
                                   # blend the result with the original image using a blobby mask
                                   iaa.SimplexNoiseAlpha(iaa.OneOf([
                                       iaa.EdgeDetect(alpha=(0.2, alpha)),
                                       iaa.DirectedEdgeDetect(alpha=(0.2, alpha), direction=(0.0, 1.0)),
                                   ])),
                                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                                   # add gaussian noise to images
                                   iaa.OneOf([
                                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                       # randomly remove up to 10% of the pixels
                                       iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                                   ]),
                                   iaa.Invert(0.05, per_channel=True),  # invert color channels
                                   iaa.Add((-10, 10), per_channel=0.5),
                                   # change brightness of images (by -10 to 10 of original value)
                                   iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                                   # either change the brightness of the whole image (sometimes
                                   # per channel) or change the brightness of subareas
                               ])
                ])

        elif augment == 4:
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
                    iaa.SomeOf((0, 7),
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
        return len(self.gt_files)

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

        if img.shape[0:2] != (self.tile_size,) * 2 or img.shape[0:2] != (self.tile_size,) * 2:
            img = cv2.resize(img, dsize=(self.tile_size,) * 2)
            gt = cv2.resize(gt, dsize=(self.tile_size,) * 2)

        if self.augment:
            img = self.seq.augment_image(img)

        # scale between 0 and 1 and swap the dimension
        img = img.transpose(2, 0, 1) / 255.0

        # normalised img between -1 and 1
        img = [np.expand_dims((img - 0.5) / 0.5, axis=0) for img in img]
        img = np.concatenate(img, axis=0)

        # get label from gt
        label = mode(gt[np.logical_and(gt > 0, gt != 4)], axis=None)[0]
        label = label if label.size > 0 else 0  # if no glands, give 0 label

        if label == 3:
            label = 1  # 3 classes need labels from 0 to 2 in loss computation

        # convert to torch tensor
        img = torch.from_numpy(img).type(torch.float)
        label = torch.tensor(label).type(torch.long).squeeze()

        return img, label


class CK5GlandDataset(Dataset):

    def __init__(self, dir_, mode, tile_size, augment=False):
        r"Dataset for feeding glands instances to network"
        self.mode = mode
        self.dir = dir_
        self.augment = augment
        self.tile_size = tile_size

        #Read data paths (images containing full glands)
        self.gt_files = glob.glob(os.path.join(self.dir, mode, '**','gland_gt_[0-9].png'), recursive=True)
        self.gt_files.extend(glob.glob(os.path.join(self.dir, mode, '**', 'gland_gt_[0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(self.dir, mode, '**', 'gland_gt_[0-9][0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(self.dir, mode, '**', 'gland_gt_[0-9][0-9][0-9][0-9].png'), recursive=True))
        assert(self.gt_files); "Cannot be empty"
        self.img_files = [re.sub('gt', 'img', gtfile) for gtfile in self.gt_files]
        assert (self.img_files)

        #Check paths
        path_check = zip(self.gt_files, self.img_files)
        for idx, paths in enumerate(path_check):
            for path in paths:
                if not is_pathname_valid(path):
                    warnings.warn("Invalid path {} was removed".format(self.gt_files[idx]))
                    del self.gt_files[idx]

        assert(self.gt_files); r"Cannot be empty"
        assert(self.img_files)

        if self.augment:
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
                                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                       # randomly remove up to 10% of the pixels
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
                                   iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                                   # improve or worsen the contrast
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
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        gt_name = self.gt_files[idx]

        bgr_img = cv2.imread(img_name, -1)
        b, g, r = cv2.split(bgr_img)  # get b,g,r

        img = cv2.merge([r, g, b])  # switch it to rgb
        gt = cv2.imread(gt_name, -1)

        if img.shape[0:2] != (self.tile_size,)*2 or img.shape[0:2] != (self.tile_size,)*2:
            img = cv2.resize(img, dsize=(self.tile_size,)*2)
            gt = cv2.resize(gt, dsize=(self.tile_size,)*2)

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

        if label == 3:
            label = 1  # 3 classes need labels from 0 to 2 in loss computation

        # convert to torch tensor
        img = torch.from_numpy(img).type(torch.float)
        label = torch.tensor(label).type(torch.long).squeeze()

        return img, label


class ERGGlandDataset(Dataset):

    def __init__(self, data_file, mode, tile_size, augment=False):
        r"Dataset for feeding glands instances to network"
        self.mode = mode
        self.augment = augment
        self.tile_size = tile_size
        img_files = []
        erg_labels = []
        with open(data_file, 'r') as glands_ERG_file:
            rd = csv.reader(glands_ERG_file, delimiter=' ')
            for row in rd:
                if mode in row[0]:
                    img_files.append(row[0])
                    erg_labels.append(int(row[1]))
        self.img_files = img_files
        self.erg_labels = erg_labels

        if self.augment:
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
                                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                       # randomly remove up to 10% of the pixels
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
                                   iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                                   # improve or worsen the contrast
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
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        label = self.erg_labels[idx]

        bgr_img = cv2.imread(img_name, -1)
        b, g, r = cv2.split(bgr_img)  # get b,g,r

        img = cv2.merge([r, g, b])  # switch it to rgb

        if img.shape[0:2] != (self.tile_size,)*2 or img.shape[0:2] != (self.tile_size,)*2:
            img = cv2.resize(img, dsize=(self.tile_size,)*2)

        if self.augment:
            img = self.seq.augment_image(img)

        # scale between 0 and 1 and swap the dimension
        img = img.transpose(2, 0, 1)/255.0

        # normalised img between -1 and 1
        img = [np.expand_dims((img - 0.5)/0.5, axis=0) for img in img]
        img = np.concatenate(img, axis=0)

        # convert to torch tensor
        img = torch.from_numpy(img).type(torch.float)
        label = torch.tensor(label).type(torch.long).squeeze()
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
        elif w < tw or h < th:
            raise ValueError("Desired dim ({}x{}) are larger than image dims ({}x{})".format(th, tw, h, w))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if img.ndim == 3:
            img = img[y1:y1+th, x1:x1+tw, :]
        else:
            img = img[y1:y1 + th, x1:x1 + tw]
        return img