import glob
import os, sys
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

sys.path.append("../mymodel")
from utils import is_pathname_valid


class GlandDataset(Dataset):

    def __init__(self, dir, mode, grayscale=False, augment=False, load_wm=False):
        r"Dataset for feeding glands instances to network"
        self.mode = mode
        self.dir = dir
        self.grayscale = grayscale
        self.augment = augment
        self.load_wm = load_wm

        #Read data paths (images containing full glands)
        self.gt_files = glob.glob(os.path.join(dir, mode, '**','gland_gt_[0-9].png'), recursive=True)
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**', 'gland_gt_[0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**', 'gland_gt_[0-9][0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**', 'gland_gt_[0-9][0-9][0-9][0-9].png'), recursive=True))
        assert(self.gt_files); "Cannot be empty"
        self.img_files = [re.sub('gt', 'img', gtfile) for gtfile in self.gt_files]
        assert (self.img_files)

        #Check paths
        path_check = zip(self.gt_files, self.img_files, self.wm_files) if self.load_wm else \
                        zip(self.gt_files, self.img_files)
        for idx, paths in enumerate(path_check):
            for path in paths:
                if not is_pathname_valid(path):
                    warnings.warn("Invalid path {} was removed".format(self.gt_files[idx]))
                    del self.gt_files[idx]

        assert(self.gt_files); r"Cannot be empty"
        assert(self.img_files)

        if self.augment:
            geom_augs = [#iaa.Affine(rotate=(-45, 45)), #keep glands aligned to long dimension
                        iaa.Affine(shear=(-10, 10)),
                        iaa.Fliplr(0.9),
                        iaa.Flipud(0.9),
                        iaa.PiecewiseAffine(scale=(0.01, 0.04)),
                        ]
            img_augs = [iaa.AdditiveGaussianNoise(scale=0.05*255),
                            iaa.Add(20, per_channel=True),
                            iaa.Dropout(p=(0, 0.2)),
                            iaa.ElasticTransformation(alpha=(0, 0.2), sigma=0.25),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.MedianBlur(k=(3, 5)),
                            iaa.Sharpen(alpha=(0.0, 0.3), lightness=(0.9, 1.1)),
                            iaa.Emboss(alpha=(0.0, 0.3), strength=(0.1, 0.3)),
                            iaa.EdgeDetect(alpha=(0.0, 0.3))]
            if not self.grayscale:
                #Add color modifications
                img_augs.append(iaa.WithChannels(0, iaa.Add((5, 20))))
                img_augs.append(iaa.WithChannels(1, iaa.Add((5, 10))))
                img_augs.append(iaa.WithChannels(2, iaa.Add((5, 20))))
                #img_augs.append(iaa.ContrastNormalization((0.8, 1.0)))
                img_augs.append(iaa.Multiply((0.9, 1.1)))
                #img_augs.append(iaa.Invert(0.5))
                img_augs.append(iaa.Grayscale(alpha=(0.0, 0.1)))
            self.geom_aug_seq = iaa.SomeOf((0, None), geom_augs) #apply 0-all augmenters; both img and gt
            self.img_seq = iaa.SomeOf((0, None), img_augs) #only img

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        try:
            img = imageio.imread(self.img_files[idx])
        except ValueError as err:
            print("#--------> Invalid img data for path: {}".format(self.img_files[idx]))
            raise err
        # try:
        #     gt = imageio.imread(self.gt_files[idx])
        # except ValueError as err:
        #     print("#--------> Invalid gt data for path: {}".format(self.img_files[idx]))
        #     raise err
        if self.load_wm:
            try:
                wm = imageio.imread(self.wm_files[idx]) / 255 #binarize
                wm = np.expand_dims(wm[:,:,0], 2)
            except ValueError as err:
                print("#--------> Invalid wm data for path: {}".format(self.img_files[idx]))
                raise err
        #assert(len(set(gt.flatten())) <= 3); "Number of classes is greater than specified"

        #Transform gt to desired number of classes
        #if self.num_class > 1: gt = self.split_gt(gt)
        #else: gt=(gt>0)[:,:,np.newaxis].astype(np.uint8)

        #Grascale
        if self.grayscale: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.augment:
            geom_seq_det = self.geom_aug_seq.to_deterministic() #ensure ground truth and image are transformed identically
            img = geom_seq_det.augment_image(img)
            img = self.img_seq.augment_image(img)
            #gt = geom_seq_det.augment_image(gt)
            if self.load_wm:
                wm = geom_seq_det.augment_image(wm)
            #print("THE SHAPE OF WM IS {}".format(wm.shape))

        img.clip(0, 255)  # ensure added values don't stray from normal boundaries
        img = img / 255  # normalize from 0 to 1

        example = self.to_tensor(img, isimage=True) #,  self.to_tensor(gt, isimage=False)
        if self.load_wm:
            example += (self.to_tensor(wm, isimage=False),)
        return example

    def split_gt(self, gt, cls_values=[0,2,4], merge_cls={4:2}):
        cls_gt=[]

        if len(cls_values) == 2: #binarize
            gt = (gt > 0).astype(np.uint8)
            cls_values=[0,1]

        #Build one gt image per class
        for c in cls_values:
            map = np.array(gt == c, dtype=np.uint8)  #simple 0 or 1 for different classes
            #could weight pixels here
            cls_gt.append(map)

        #Handle overlapping classes (fill)
        for cs, ct in merge_cls.items():
            mask = cls_gt[cls_values.index(cs)] > 0
            cls_gt[cls_values.index(ct)][mask] = 1

        gt = np.stack(cls_gt, axis=2) #need to rescale with opencv later, so channel must be last dim
        return gt

    def to_tensor(self, na, isimage):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        na = na[np.newaxis,:,:] if self.grayscale and isimage else na.transpose((2, 0, 1)) # grayscale or RGB
        na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
        return na


class GlandPatchDataset(Dataset):

    def __init__(self, dir, mode, tile_size=512, augment=False, return_cls=False):

        self.mode = mode
        self.dir = dir
        self.tile_size = tile_size
        self.augment = augment
        self.return_cls = return_cls

        # Read data paths (images containing gland parts)
        self.gt_files = glob.glob(os.path.join(dir, mode, '**','gland_gt_[0-9]_([0-9],[0-9]).png'), recursive=True)
        n = "[0-9]"
        for gl_idx, x, y in product(range(1,4), range(1,5), range(1,5)):
            to_glob = os.path.join(dir, mode, '**', 'gland_gt_' + n*gl_idx + '_(' + n*x + ',' + n*y + ').png')
            self.gt_files += glob.glob(to_glob, recursive=True)

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

        if self.augment:
            alpha = 0.2
            self.seq = iaa.Sequential(
                [
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.8),  # horizontally flip
                    iaa.Flipud(0.8),  # vertically flip
                    # crop images by -5% to 10% of their height/width
                    # sometimes(iaa.CropAndPad(
                    #     percent=(-0.05, 0.1),
                    #     pad_mode=ia.ALL,
                    #     pad_cval=(0, 255)
                    # )),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45),  # rotate by -45 to +45 degrees
                        shear=(-2, 2),  # shear by -16 to +16 degrees
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                        mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    ),
                    # execute 0 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 5),
                               [  # convert images into their superpixel representation
                                   # iaa.WithChannels([0, 1, 2],
                                   #                  iaa.OneOf([
                                   #                      iaa.GaussianBlur((0, 3.0)),
                                   #                      # blur images with a sigma between 0 and 3.0
                                   #                      iaa.AverageBlur(k=(2, 7)),
                                   #                      # blur image using local means with kernel sizes between 2 and 7
                                   #                      iaa.MedianBlur(k=(3, 11)),
                                   #                      # blur image using local medians with kernel sizes between 2 and 7
                                   #                  ])),
                                   iaa.WithChannels([0, 1, 2], iaa.Sharpen(alpha=(0, alpha), lightness=(0.75, 1.5))),
                                   # sharpen images
                                   iaa.WithChannels([0, 1, 2], iaa.Emboss(alpha=(0, alpha), strength=(0, 0.1))),  # emboss images
                                   # search either for all edges or for directed edges,
                                   # blend the result with the original image using a blobby mask
                                   iaa.WithChannels([0, 1, 2], iaa.SimplexNoiseAlpha(iaa.OneOf([
                                       iaa.EdgeDetect(alpha=(0.05, alpha)),
                                       iaa.DirectedEdgeDetect(alpha=(0.05, alpha), direction=(0.0, 1.0)),
                                   ]))),
                                   iaa.WithChannels([0, 1, 2],
                                                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
                                   # add gaussian noise to images
                                   # iaa.WithChannels([0, 1, 2], #iaa.OneOf([
                                   # iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2)])),
                                   iaa.WithChannels([0, 1, 2], iaa.Invert(0.05, per_channel=True)),  # invert color channels
                                   iaa.WithChannels([0, 1, 2], iaa.Add((-10, 10), per_channel=0.5)),
                                   # change brightness of images (by -10 to 10 of original value)
                                   iaa.WithChannels([0, 1, 2], iaa.AddToHueAndSaturation((-1, 1))),  # change hue and saturation
                                   # either change the brightness of the whole image (sometimes
                                   # per channel) or change the brightness of subareas
                                   iaa.WithChannels([0, 1, 2], iaa.OneOf([
                                       iaa.Multiply((0.8, 1.3), per_channel=0.5),
                                       iaa.FrequencyNoiseAlpha(
                                           exponent=(-2, 2),
                                           first=iaa.Multiply((0.8, 1.3), per_channel=True),
                                           second=iaa.ContrastNormalization((0.5, 2.0)))
                                   ])),
                                   iaa.WithChannels([0, 1, 2], iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
                                   # improve or worsen the contrast
                                   iaa.WithChannels([0, 1, 2], iaa.Grayscale(alpha=(0.0, alpha))),
                                   # move pixels locally around (with random strengths)
                                   # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                                   # sometimes move parts of the image around
                                   # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                               ],
                               random_order=True
                               )
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

        if self.return_cls:
            tumour_cls = stats.mode(gt[np.logical_and(gt > 0, gt != 4)], axis=None)[0]  # take most common class over gland excluding lumen
            tumour_cls = int(tumour_cls) if tumour_cls.size > 0 else 0

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

            # if x_tile > w_origin // 2:
            #     left = (self.tile_size - img.shape[1])
            #     right = 0
            # else:
            #     right = self.tile_size - img.shape[1]
            #     left = 0
            #
            # if y_tile > h_origin // 2:
            #     bottom = self.tile_size - img.shape[0]
            #     top = 0
            # else:
            #     top = self.tile_size - img.shape[0]
            #     bottom = 0

            left, right, bottom, top = max(0, left), max(0, right), max(0, bottom), max(0, top)

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=stromal_mean)
            gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

        img[gt == 0, 0] = stromal_mean[0]
        img[gt == 0, 1] = stromal_mean[1]
        img[gt == 0, 2] = stromal_mean[2]

        if img.shape[0] > self.tile_size or img.shape[1] > self.tile_size:
            img = cv2.resize(img, (self.tile_size,)*2, interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (self.tile_size,)*2, interpolation=cv2.INTER_NEAREST)

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

        data = (img, gt, torch.tensor(colour + [size]).float())
        if self.return_cls:
            data += (tumour_cls,)  # batch of these is automatically turned into a tensor by dataloader !

        return data

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



















