import glob
import os
import re
import cv2
import random
import numbers
from pathlib import Path
from itertools import product

import imageio
import torch
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision.transforms import Compose
from torch.utils.data import Dataset

ia.seed(1)


class NucleiDataset(Dataset):

    def __init__(self, dir_, mode, tile_size=512, augment=0):
        self.mode = mode
        self.augment = augment
        self.tile_size = tile_size

        self.image_files = []
        self.label = []

        thedir = os.path.join(dir_, mode)
        folders = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

        image_files, label_files = [], []
        for fold in folders:
            image_files += glob.glob(os.path.join(thedir, fold, 'images', '*.png'))
            label_files += glob.glob(os.path.join(thedir, fold, 'masks_001', '*_gt.png'))

        self.image_files = image_files
        self.label_files = label_files
        assert self.image_files
        assert self.label_files
        assert len(self.image_files) == len(self.label_files)

        self.dir = dir_

        self.randomcrop = RandomCrop(1024)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        if augment == 1:
            self.seq = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                    iaa.Flipud(0.2),  # vertically flip 20% of all images
                    # crop images by -5% to 10% of their height/width
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=["reflect", "symmetric"],
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"]  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
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
                        pad_mode=["reflect", "symmetric"],
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"]  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 2),
                               [
                                   sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                                   iaa.OneOf([
                                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                       # randomly remove up to 10% of the pixels
                                       iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                                   ]),
                                   # convert images into their superpixel representation
                                   iaa.OneOf([
                                       iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                       iaa.AverageBlur(k=(2, 7)),
                                       # blur image using local means with kernel sizes between 2 and 7
                                       iaa.MedianBlur(k=(3, 11)),
                                       # blur image using local medians with kernel sizes between 2 and 7
                                   ]),
                                   # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                                   # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                                   # # search either for all edges or for directed edges,
                                   # # blend the result with the original image using a blobby mask
                                   # iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                                   # ]))
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
                        pad_mode=["reflect", "symmetric"],
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"]  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.WithChannels([0, 1, 2],
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
                                                    # iaa.Sharpen(alpha=(0, alpha), lightness=(0.75, 1.5)),  # sharpen images
                                                    # iaa.Emboss(alpha=(0, alpha), strength=(0, 2.0)),  # emboss images
                                                    # # search either for all edges or for directed edges,
                                                    # # blend the result with the original image using a blobby mask
                                                    # iaa.SimplexNoiseAlpha(iaa.OneOf([
                                                    #     iaa.EdgeDetect(alpha=(0.2, alpha)),
                                                    #     iaa.DirectedEdgeDetect(alpha=(0.2, alpha), direction=(0.0, 1.0)),
                                                    # ])),
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
                                                ]))
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
                        pad_mode=["reflect", "symmetric"],
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-16, 16),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"]  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.WithChannels([0, 1, 2],
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
                                                    #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images  # { REMOVED AS NOT WORKING ON MULTIPROCESSING https://github.com/aleju/imgaug/issues/147
                                                    #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                                                    # search either for all edges or for directed edges,
                                                    # blend the result with the original image using a blobby mask
                                                    # iaa.SimplexNoiseAlpha(iaa.OneOf([
                                                    #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                                    #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                                                    # ])),                                                                  # }
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
                                                    #iaa.Grayscale(alpha=(0.0, 1.0)),
                                                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                                                    # move pixels locally around (with random strengths)
                                                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                                                    # sometimes move parts of the image around
                                                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                                                ],
                                                random_order=True
                                                ))
                ],
                random_order=True
            )

    def __len__(self):
        return len(self.image_files)

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        # invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        gt_name = self.label_files[idx]

        bgr_img = cv2.imread(img_name, -1)
        b, g, r = cv2.split(bgr_img)  # get b,g,r

        image = cv2.merge([r, g, b])  # switch it to rgb
        gt = cv2.imread(gt_name, -1)
        if not (isinstance(gt, np.ndarray) and gt.ndim > 0):
            raise ValueError("{} is not valid".format(gt_name))

        if gt.ndim == 3 and gt.shape[2] == 3:
            gt = gt[..., 0]
        gt[gt > 0] = 255

        if image.shape[0:2] != (self.tile_size,)*2:
            too_narrow = image.shape[1] < 1024
            too_short = image.shape[0] < 1024
            if too_narrow or too_short:
                delta_w = 1024 - image.shape[1] if too_narrow else 0
                delta_h = 1024 - image.shape[0] if too_short else 0
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
                gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_REFLECT)

            if image.shape[0] > 1024 or image.shape[1] > 1024:
                cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
                cat = self.randomcrop(cat)
                image = cat[:, :, 0:3]
                gt = cat[:, :, 3]


            # scale image
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        # im aug
        cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
        if self.augment:
            cat = self.seq.augment_image(cat)
        image = cat[:, :, 0:3]
        gt = cat[:, :, 3]
        gt[gt < 255] = 0

        # scale between 0 and 1 and swap the dimension
        image = image.transpose(2, 0, 1)/255.0
        gt = np.expand_dims(gt, axis=2).transpose(2, 0, 1)/255.0

        # normalised image between -1 and 1
        image = [np.expand_dims((img - 0.5)/0.5, axis=0) for img in image]
        image = np.concatenate(image, axis=0)

        # convert to torch tensor
        dtype = torch.FloatTensor
        image = torch.from_numpy(image).type(dtype)
        gt = torch.from_numpy(gt).type(dtype) #change to FloatTensor for BCE

        return image, gt


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