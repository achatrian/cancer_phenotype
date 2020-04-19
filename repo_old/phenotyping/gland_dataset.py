import glob
import os
import sys
import re
import cv2
import random
import numbers
from collections import namedtuple
from pathlib import Path

import imageio
import torch
import numpy as np
from scipy import stats
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision.transforms import Compose
from torch.utils.data import Dataset
import warnings

from itertools import product

def on_cluster():
    import socket, re
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)

if on_cluster():
    sys.path.append("/gpfs0/users/rittscher/achatrian/cancer_phenotype")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype")
from segment.utils import is_pathname_valid


class GlandDataset(Dataset):

    def __init__(self, dir_, mode, tile_size, augment=False, blur=False, return_path=False):
        r"Dataset for feeding glands instances to network"
        self.mode = mode
        self.dir = dir_
        self.augment = augment
        self.blur = blur
        self.tile_size = tile_size
        self.return_path = return_path

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

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        if self.augment:
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
                        mode=["reflect", "symmetric"]
                        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    )),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.WithChannels([0, 1, 2],
                                     iaa.SomeOf((0, 7),
                                                [
                                                    sometimes(
                                                        iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                                                    # convert images into their superpixel representation
                                                    iaa.OneOf([
                                                        iaa.GaussianBlur((0, 3.0)),
                                                        # blur images with a sigma between 0 and 3.0
                                                        iaa.AverageBlur(k=(2, 7)),
                                                        # blur image using local means with kernel sizes between 2 and 7
                                                        iaa.MedianBlur(k=(3, 11)),
                                                        # blur image using local medians with kernel sizes between 2 and 7
                                                    ]),
                                                    # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images  # { REMOVED AS NOT WORKING ON MULTIPROCESSING https://github.com/aleju/imgaug/issues/147
                                                    # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                                                    # search either for all edges or for directed edges,
                                                    # blend the result with the original image using a blobby mask
                                                    # iaa.SimplexNoiseAlpha(iaa.OneOf([
                                                    #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                                    #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                                                    # ])),                                                                  # }
                                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255),
                                                                              per_channel=0.5),
                                                    # add gaussian noise to images
                                                    iaa.OneOf([
                                                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                                        # randomly remove up to 10% of the pixels
                                                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05),
                                                                          per_channel=0.2),
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
                                                    # iaa.Grayscale(alpha=(0.0, 1.0)),
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
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        gt_path = self.gt_files[idx]

        bgr_img = cv2.imread(img_path, -1)
        b, g, r = cv2.split(bgr_img)  # get b,g,r

        img = cv2.merge([r, g, b])  # switch it to rgb
        gt = cv2.imread(gt_path, -1)

        if img.shape[0:2] != (self.tile_size,)*2 or img.shape[0:2] != (self.tile_size,)*2:
            img = cv2.resize(img, dsize=(self.tile_size,)*2)
            gt = cv2.resize(gt, dsize=(self.tile_size,)*2)

        # get label from gt
        tumour_cls = stats.mode(gt[np.logical_and(gt > 0, gt != 4)], axis=None)[0]
        tumour_cls = tumour_cls if tumour_cls.size > 0 else 0  # if no glands, give 0 label

        colour, size = self.gland_colour_size(img, gt)

        if tumour_cls == 3:
            tumour_cls = 1  # 3 classes need labels from 0 to 2 in loss computation

        # Normalize as wh en training network:
        if self.augment:
            gt = gt[..., np.newaxis]
            cat = np.concatenate([img, gt], axis=2)
            cat = self.seq.augment_image(cat)
            img = cat[..., 0:3]
            gt = cat[..., 3]
            img = img.clip(0, 255)
        if self.blur:
            img = cv2.bilateralFilter(img, 5, 75, 75)
        img = img / 255  # from 0 to 1
        img = (img - 0.5)/0.5  # from -1 to 1

        assert img.max() <= 1.0
        assert img.min() >= -1.0

        gt = gt[:, :, np.newaxis]

        img = self.to_tensor(img)
        gt = self.to_tensor(gt)  # NB dataloader must return tensors

        col_size = torch.tensor(colour + [size]).float()
        tumour_cls = torch.tensor(tumour_cls).long().squeeze()

        assert tuple(img.shape) == (3, self.tile_size, self.tile_size)
        assert tuple(gt.shape) == (1, self.tile_size, self.tile_size)
        assert col_size.shape[0] == 4
        assert not bool(tumour_cls.shape)  # should be empty

        return img, gt, col_size, tumour_cls, img_path

    def __len__(self):
        return len(self.gt_files)

    @staticmethod
    def to_tensor(na):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        na = na.transpose(2, 0, 1)
        na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
        return na

    @staticmethod
    def gland_colour_size(img, gt):
        """
        :param img:
        :param gt:
        :return:
        """
        EPS = 0.1
        gt = gt.squeeze()
        with np.errstate(divide='ignore'):
            try:
                colours = [int(cc[np.logical_or(np.isclose(gt, 2), np.isclose(gt, 3))].mean()) for cc in img.transpose(2, 0, 1)]
            except ValueError:
                colours = [cc.mean() for cc in img.transpose(2, 0, 1)]
        size = np.sqrt(np.sum(gt > (0 + EPS)))
        return colours, size

#kernel = np.ones((3, 3), np.uint8)
class GTDataset(Dataset):

    def __init__(self, dir, mode, tile_size=256, augment=False):
        "GTs only for vae"
        self.mode = mode
        self.dir = dir
        self.tile_size = tile_size
        self.augment = augment

        # Read data paths (images containing gland parts)
        self.gt_files = glob.glob(os.path.join(dir, mode, '**','gland_img_[0-9]_([0-9],[0-9]).png'), recursive=True)
        n = "[0-9]"
        for gl_idx, x, y in product(range(1, 8), range(1, 6), range(1, 6)):
            to_glob = os.path.join(dir, mode, '**', 'gland_img_' + n*gl_idx + '_(' + n*x + ',' + n*y + ').png')
            self.gt_files += glob.glob(to_glob, recursive=True)

        assert(self.gt_files); r"Cannot be empty"

        # Check paths
        path_check = self.gt_files
        for idx, paths in enumerate(path_check):
            for path in paths:
                if not is_pathname_valid(path):
                    warnings.warn("Invalid path {} was removed".format(self.gt_files[idx]))
                    del self.gt_files[idx]

        assert(self.gt_files); r"All paths invalid"

        if self.augment:
            self.seq = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.8),  # horizontally flip
                    iaa.Flipud(0.8),  # vertically flip
                    # crop images by -5% to 10% of their height/width
                    iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=["reflect", "symmetric"],
                    ),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-2, 2),  # shear by -16 to +16 degrees
                        order=1,  # use nearest neighbour or bilinear interpolation (fast)
                        mode=["reflect", "symmetric"],  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    ),
                ],
                random_order=True
            )

    def __getitem__(self, idx):
        try:
            gt = imageio.imread(self.gt_files[idx])
        except ValueError as err:
            print("#--------> Invalid gt data for path: {}".format(self.img_files[idx]))
            raise err

        gt = gt[:, :gt.shape[0], :]  # only second half is gt

        if gt.shape[0:2] != (self.tile_size,)*2:
            gt = cv2.resize(gt, dsize=(self.tile_size,)*2)


        label = stats.mode(gt[np.logical_and(gt > 0, gt != 250)], axis=None)[0]  # take most common class over gland excluding lumen
        if label.size > 0:
            label = int(label)
            if label == 160:
                label = 0
            elif label == 200:
                label = 1
        else:
            label = 0.5

        gt[gt == 160] = 40  # to help get better map with irregularities introduced by augmentation

        # Normalize as wh en training network:

        gt = gt[:, :, 0]
        gt = np.stack((np.uint8(np.logical_and(gt >= 0, gt < 35)),
                         np.uint8(np.logical_and(gt >= 35, gt < 45)),
                         np.uint8(np.logical_and(gt >= 194, gt < 210)),
                         np.uint8(np.logical_and(gt >= 210, gt <= 255))), axis=2)

        if self.augment:
            gt = self.seq.augment_image(gt)
            gt = gt.clip(0, 1)
            #gt = cv2.morphologyEx(gt, cv2.MORPH_OPEN, kernel)
            #gt = cv2.dilate(gt, kernel, iterations=1)

        assert gt.max() <= 1.0
        assert gt.min() >= 0.0

        gt = self.to_tensor(gt)  # NB dataloader must return tensors
        label = torch.tensor(label).float().squeeze()
        assert not bool(label.shape)  # should be empty

        return gt, label, self.gt_files[idx]

    def __len__(self):
        return len(self.gt_files)

    @staticmethod
    def to_tensor(na):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        na = na.transpose(2, 0, 1)
        na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
        return na

def gt_to_img(gt):
    gt_1d = np.uint8(gt[:, :, 3] >= 0.5) * 250
    gt_1d[np.logical_and(gt[:, :, 3] < 0.5, gt[:, :, 2] >= 0.5)] = 200
    gt_1d[np.logical_and(gt[:, :, 3] < 0.5, gt[:, :, 1] >= 0.5)] = 160
    return gt_1d[:, :, np.newaxis].repeat(3, axis=2)



class GlandPatchDataset(Dataset):

    def __init__(self, dir_, mode, tile_size=512, augment=0, return_cls=False):

        self.mode = mode
        self.dir = dir_
        self.tile_size = tile_size
        self.augment = augment
        self.return_cls = return_cls

        # Read data paths (images containing gland parts)
        thedir = os.path.join(dir_, mode, "glands_full")
        folders = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]
        self.gt_files = glob.glob(os.path.join(dir_, mode, '**','gland_gt_[0-9]_([0-9],[0-9]).png'), recursive=True)
        n = "[0-9]"
        for folder in folders:
            for gl_idx, x, y in product(range(1,7), range(1,8), range(1,8)):
                to_glob = os.path.join(thedir, folder, 'gland_gt_' + n*gl_idx + '_(' + n*x + ',' + n*y + ').png')
                self.gt_files += glob.glob(to_glob)

        assert(self.gt_files); r"Cannot be empty"
        self.img_files = [re.sub('gt', 'img', gtfile) for gtfile in self.gt_files]
        assert (self.img_files)

        # Check paths
        path_check = zip(self.gt_files, self.img_files)
        for idx, paths in enumerate(path_check):
            for path in paths:
                if not is_pathname_valid(path):
                    warnings.warn("Invalid path {} was removed".format(self.gt_files[idx]))
                    del self.gt_files[idx]

        assert(self.gt_files); r"All paths invalid"
        assert(self.img_files)

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

    def __getitem__(self, idx):
        try:
            img = imageio.imread(self.img_files[idx])
        except ValueError as err:
            print("#--------> Invalid img data for path: {}".format(self.img_files[idx]))
            raise err
        try:
            gt = imageio.imread(self.gt_files[idx])
        except ValueError as err:
            print("#--------> Invalid gt data for path: {}".format(self.img_files[idx]))
            raise err

        tumour_cls = stats.mode(gt[np.logical_and(gt > 0, np.logical_not(np.isclose(gt, 250, atol=10)))], axis=None)[0] # take most common class over gland excluding lumen
        tumour_cls = int(tumour_cls) if tumour_cls.size > 0 else 0

        if np.isclose(tumour_cls, 160, atol=10):
            tumour_cls = 2
        elif np.isclose(tumour_cls, 200, atol=10):
            tumour_cls = 3

        if gt.ndim == 3:  # for augmented gts
            gt = gt[..., 0]
        colour, size = self.gland_colour_size(img, gt)

        gt[gt > 0] = 1  # push to one class

        bg_mask = np.isclose(gt.squeeze(), 0)  # background mask
        stromal_mean = [np.mean(img[bg_mask, 0]), np.mean(img[bg_mask, 1]), np.mean(img[bg_mask, 1])]
        if img.shape[0] < self.tile_size or img.shape[1] < self.tile_size:
            # img_dir = Path(self.img_files[idx]).parent
            # w_origin, h_origin = str(img_dir)[:-1].split('(')[1].split(',')[-2:]  # get original img size from folder name
            # x_tile, y_tile = self.img_files[idx][:-5].split('/')[-1].split('(')[1].split(',')  # get position of gland tile relative to larger image
            # w_origin, h_origin, x_tile, y_tile = [int(s) for s in [w_origin, h_origin, x_tile, y_tile]]

            left = (self.tile_size - img.shape[1]) // 2
            right = (self.tile_size - img.shape[1]) - left
            bottom = (self.tile_size - img.shape[0]) // 2
            top = (self.tile_size - img.shape[0]) - bottom

            left, right, bottom, top = max(0, left), max(0, right), max(0, bottom), max(0, top)

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=stromal_mean)
            gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

        img[gt == 0, 0] = stromal_mean[0]
        img[gt == 0, 1] = stromal_mean[1]
        img[gt == 0, 2] = stromal_mean[2]

        if img.shape[0] > self.tile_size or img.shape[1] > self.tile_size:
            img = cv2.resize(img, (self.tile_size,)*2, interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (self.tile_size,)*2, interpolation=cv2.INTER_CUBIC)

        gt = gt[:, :, np.newaxis]

        # Normalize as wh en training network:
        if self.augment:
            cat = np.concatenate([img, gt], axis=2)
            cat = self.seq.augment_image(cat)
            img = cat[..., 0:3]
            gt = cat[..., 3]
            gt = gt[:, :, np.newaxis]
            img = img.clip(0, 255)


        img = img / 255  # from 0 to 1
        img = (img - 0.5)/0.5  # from -1 to 1

        assert img.max() <= 1.0
        assert img.min() >= -1.0

        img = self.to_tensor(img)
        gt = self.to_tensor(gt)  # NB dataloader must return tensors

        col_size = torch.tensor(colour + [size]).float()
        tumour_cls = torch.tensor(tumour_cls).long().squeeze()

        assert tuple(img.shape) == (3, self.tile_size, self.tile_size)
        assert tuple(gt.shape) == (1, self.tile_size, self.tile_size)
        assert col_size.shape[0] == 4
        assert not bool(tumour_cls.shape)  # should be empty

        return img, gt, col_size, tumour_cls, self.img_files[idx]

    def __len__(self):
        return len(self.gt_files)

    @staticmethod
    def to_tensor(na):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        na = na.transpose(2, 0, 1)
        na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
        return na

    @staticmethod
    def gland_colour_size(img, gt):
        """
        :param img:
        :param gt:
        :return:
        """
        EPS = 0.1
        gt = gt.squeeze()
        with np.errstate(divide='ignore'):
            try:
                colours = [int(cc[np.logical_or(np.isclose(gt, 2), np.isclose(gt, 3))].mean()) for cc in img.transpose(2, 0, 1)]
            except ValueError:
                colours = [cc.mean() for cc in img.transpose(2, 0, 1)]
        size = np.sqrt(np.sum(gt > (0 + EPS)))
        return colours, size


class AugDataset(GlandPatchDataset):

    def __init__(self, dir_, aug_dir, mode, tile_size, augment=0, generated_only=True):
        super(AugDataset, self).__init__(dir_, mode, tile_size, augment)
        if generated_only:
            self.file_list = []
            self.label = []
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















