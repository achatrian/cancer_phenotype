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
            tumour_cls = [stats.mode(gt[np.logical_and(gt > 0, gt != 4)], axis=None)[0] for gt in gt] # take most common class over gland excluding lumen
            tumour_cls = [int(tc) if tc.size > 0 else 0 for tc in tumour_cls]
        gt[gt > 0] = 1  # push to one class

        stromal_mean = np.mean(img[np.isclose(gt, 0)])
        if img.shape[0] < self.tile_size or img.shape[1] < self.tile_size:
            img_dir = Path(self.img_files[idx]).parent
            w_origin, h_origin = str(img_dir)[:-1].split('(')[1].split(',')[-2:]  # get original img size from folder name
            x_tile, y_tile = self.img_files[idx][:-5].split('/')[-1].split('(')[1].split(',')  # get position of gland tile relative to larger image
            w_origin, h_origin, x_tile, y_tile = [int(s) for s in [w_origin, h_origin, x_tile, y_tile]]

            if x_tile > w_origin // 2:
                left = self.tile_size - img.shape[1]
                right = 0
            else:
                right = self.tile_size - img.shape[1]
                left = 0

            if y_tile > h_origin // 2:
                bottom = self.tile_size - img.shape[0]
                top = 0
            else:
                top = self.tile_size - img.shape[0]
                bottom = 0

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[stromal_mean]*3)
            gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

        gt = gt[:, :, np.newaxis]
        img[gt.repeat(3, axis=2) == 0] = stromal_mean  # doing this outside as it breaks frequency augmentation

        # Normalize as when training network:
        #img = img.clip(0, 255) # NOT NEEDED
        img = img / 255  # from 0 to 1
        img = (img - 0.5)/0.5  # from -1 to 1

        assert img.max() <= 1.0
        assert img.min() >= -1.0

        img = self.to_tensor(img)
        gt = self.to_tensor(gt)  # NB dataloader must return tensors

        data = (img, gt)
        if self.return_cls:
            data += (tumour_cls,)
        return data


    def __len__(self):
        return len(self.gt_files)

    def to_tensor(self, na):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        na = na.transpose(2, 0, 1)
        na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
        return na



















