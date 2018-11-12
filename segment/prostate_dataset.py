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
import warnings

from utils import is_pathname_valid

ia.seed(1)


class SegDataset(Dataset):

    def __init__(self, dir_, mode, tile_size=256, augment=0):
        self.mode = mode
        self.augment = augment
        self.tile_size = tile_size

        self.file_list = []
        self.label = []

        thedir = os.path.join(dir_, mode)
        folders = [name for name in os.listdir(thedir) if os.path.isdir(os.path.join(thedir, name))]

        file_list = []
        for file in folders:
            file_list += glob.glob(os.path.join(thedir, file, 'tiles', '*_img_*.png'))

        self.file_list = file_list
        self.label = [x.replace('_img_', '_mask_') for x in file_list]

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
        return len(self.file_list)

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        # invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        gt_name = self.label[idx]

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


class AugDataset(SegDataset):

    def __init__(self, dir_, aug_dir, mode, tile_size, augment=0, generated_only=False):
        super(AugDataset, self).__init__(dir_, mode, tile_size, augment)
        if generated_only:
            self.file_list = []
            self.label = []
        n = "[0-9]"
        names = ["_rec1_", "_rec2_"] #, "_gen1_", "_gen2_"]
        no_aug_len = len(self.file_list)
        file_list = []
        for gl_idx, x, y, name in product(range(1, 7), range(1, 6), range(1, 6), names):
            to_glob = os.path.join(aug_dir, 'gland_img_' + n * gl_idx + '_(' + n * x + ',' + n * y + ')' + \
                                   name + 'fake_B.png')
            file_list += glob.glob(to_glob)
        self.file_list += file_list
        self.label += [x.replace('fake_B', 'real_A') for x in file_list]
        assert (len(self.file_list) > no_aug_len)


class TestDataset(SegDataset):
    def __init__(self, dir_, tile_size=256, bad_folds=[]):
        super(TestDataset, self).__init__(dir_, "test", tile_size, augment=0)
        if bad_folds:
            for image_file, label_file in zip(self.file_list, self.label):
                image_name = os.path.basename(image_file)
                label_name = os.path.basename(label_file)
                assert(image_name[0:10] == label_name[0:10])
                isbad = any([bad_fold in image_name for bad_fold in bad_folds])
                if isbad:
                    self.file_list.remove(image_file)
                    self.label.remove(label_file)



class ProstateDataset(Dataset):

    def __init__(self, dir, mode, out_size=1024, down=2.0, num_class=1, grayscale=False, augment=None, load_wm=False):
        r"""
        Dataset to return random (pre-made) tiles of WSI image at original resolution.
        Can downscale
        Make grayscale
        Augment
        Use pre-made weight map for each example
        """
        self.mode = mode
        self.dir = dir
        self.out_size = out_size
        self.down = down
        self.num_class = num_class
        self.grayscale = grayscale
        self.augment = augment #set augment, or augment when training
        self.load_wm = load_wm
        assert(mode in ['train', 'validate', 'test'])

        #Read data paths
        self.gt_files = glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9],[0-9].png'), recursive=True) #for 1,1 patch
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9],[0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9][0-9],[0-9][0-9][0-9].png'), recursive=True))
        self.gt_files.extend(glob.glob(os.path.join(dir, mode, '**','*_mask_[0-9][0-9][0-9][0-9],[0-9][0-9][0-9][0-9].png'), recursive=True))

        self.img_files = [re.sub('mask', 'img', gtfile) for gtfile in self.gt_files]
        if self.load_wm:
            self.wm_files = []
            for gtfile in self.gt_files:
                srch = re.search('_mask_([0-9\(\)]+),([0-9\(\)]+).png', gtfile)
                self.wm_files.append(str(Path(gtfile).parents[1]/"weightmaps"/"weightmap_{},{}.png".format(srch.group(1), srch.group(2))))

        #Check paths
        path_check = zip(self.gt_files, self.img_files, self.wm_files) if self.load_wm else \
                        zip(self.gt_files, self.img_files)
        for idx, paths in enumerate(path_check):
            for path in paths:
                if not is_pathname_valid(path):
                    warnings.warn("Invalid path {} was removed".format(self.gt_files[idx]))
                    del self.gt_files[idx]


        assert(self.gt_files); r"Cannot be empty"
        assert(self.img_files);

        #Augmentation sequence
        self.downsample = Downsample(down, min_size=out_size)
        if self.mode == 'train': self.crop_aug = RandomCrop(int(out_size*down))
        elif self.mode in ['validate', 'test']: self.center_crop = CenterCrop(int(out_size*down))

        if self.augment:
            geom_augs = [iaa.Affine(rotate=(-45, 45)),
                        iaa.Affine(shear=(-10, 10)),
                        iaa.Fliplr(0.9),
                        iaa.Flipud(0.9),
                        iaa.PiecewiseAffine(scale=(0.01, 0.04)),
                        ]
            img_augs = [iaa.AdditiveGaussianNoise(scale=0.05*255),
                            iaa.Add(20, per_channel=True),
                            iaa.Dropout(p=(0, 0.2)),
                            iaa.ElasticTransformation(alpha=(0, 3.0), sigma=0.25),
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
                img_augs.append(iaa.Grayscale(alpha=(0.0, 0.2)))
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
        try:
            gt = imageio.imread(self.gt_files[idx])
        except ValueError as err:
            print("#--------> Invalid gt data for path: {}".format(self.img_files[idx]))
            raise err
        if self.load_wm:
            try:
                wm = imageio.imread(self.wm_files[idx]) / 255 #binarize
                wm = np.expand_dims(wm[:,:,0], 2)
            except ValueError as err:
                print("#--------> Invalid wm data for path: {}".format(self.img_files[idx]))
                raise err
        assert(len(set(gt.flatten())) <= 3); "Number of classes is greater than specified"

        #Pad images to desired size (for smaller tiles):
        size_b4_down = self.out_size * int(self.down)
        too_narrow = img.shape[1] < size_b4_down
        too_short = img.shape[0] < size_b4_down
        if too_narrow or too_short:
            delta_w = size_b4_down - img.shape[1] if too_narrow else 0
            delta_h = size_b4_down - img.shape[0] if too_short else 0
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            #if self.load_wm:
                #wm = cv2.copyMakeBorder(wm, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            assert(img.shape[0] >= size_b4_down and img.shape[1] >= size_b4_down)  #Ensure padded
            assert(gt.shape[0] >= size_b4_down and gt.shape[1] >= size_b4_down)  #Ensure padded

        #Transform gt to desired number of classes
        if self.num_class > 1: gt = self.split_gt(gt)
        else: gt=(gt>0)[:,:,np.newaxis].astype(np.uint8)

        #Grascale
        if self.grayscale: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.mode == 'train':
            img = self.crop_aug([img], random_state=1)[0] #crop to out_size * downsample
            img = self.downsample(img) #downsample to size out_sizee
            gt = self.crop_aug([gt], random_state=1)[0]
            gt = self.downsample(gt)
            #if self.load_wm:
                #wm = self.crop_aug([wm], random_state=1)[0]
                #wm = self.downsample(wm)
        else:
            img = self.center_crop(img) #center crop tiles when testing
            img = self.downsample(img) #downsample like in training
            gt = self.center_crop(gt)
            gt = self.downsample(gt)
            #if self.load_wm:
                #wm = self.center_crop([wm])[0]
                #wm = self.downsample(wm)

        if self.augment:
            geom_seq_det = self.geom_aug_seq.to_deterministic() #ensure ground truth and image are transformed identically
            img = geom_seq_det.augment_image(img)
            img = self.img_seq.augment_image(img)
            img.clip(0, 255) #ensure added values don't stray from normal boundaries
            gt = geom_seq_det.augment_image(gt)
            if self.load_wm:
                wm = geom_seq_det.augment_image(wm)
            #print("THE SHAPE OF WM IS {}".format(wm.shape))

        example = self.to_tensor(img, isimage=True),  self.to_tensor(gt, isimage=False)
        if self.load_wm:
            example += (self.to_tensor(wm, isimage=False),)
        return example

    def split_gt(self, gt, cls_values=[0,2,4], merge_cls={4:2}):
        cls_gts=[]

        if len(cls_values) == 2: #binarize
            gt = (gt > 0).astype(np.uint8)
            cls_values=[0,1]

        #Build one gt image per class
        for c in cls_values:
            map = np.array(gt == c, dtype=np.uint8)  #simple 0 or 1 for different classes
            #could weight pixels here
            cls_gts.append(map)

        #Handle overlapping classes (fill)
        for cs, ct in merge_cls.items():
            mask = cls_gts[cls_values.index(cs)] > 0
            cls_gts[cls_values.index(ct)][mask] = 1

        gt = np.stack(cls_gts, axis=2) #need to rescale with opencv later, so channel must be last dim
        return gt

    def to_tensor(self, na, isimage):
        r"""Convert ndarrays in sample to Tensors."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        na = na[np.newaxis,:,:] if self.grayscale and isimage else na.transpose((2, 0, 1)) #grayscale or RGB
        na = torch.from_numpy(na.copy()).type(torch.FloatTensor)
        return na

######



#####################################################
#dataset utils


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

#####
class Compose(object):
    "Apply set of transforms to image"
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Scale(object):
    """Rescales the input np.ndarray to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv.INTER_CUBIC
    """
    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(float(self.size) * h / w)
        else:
            oh = self.size
            ow = int(float(self.size) * w / h)
        return cv2.resize(img, dsize=(ow, oh),
                          interpolation=self.interpolation)

class Downsample(object):
    """Downsamples the input by a given factor
    interpolation: Default: cv.INTER_CUBIC
    """
    def __init__(self, factor, min_size=None, interpolation=cv2.INTER_CUBIC):
        self.factor = float(factor)
        self.min_size = min_size or 10000
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        try:
            c = img.shape[2]
        except IndexError:
            c = 0
        ow = int(np.ceil(w / self.factor))
        oh = int(np.ceil(w / self.factor))
        ow = ow if ow > self.min_size else self.min_size
        oh = oh if oh > self.min_size else self.min_size
        outimg = cv2.resize(img, dsize=(ow, oh),
                          interpolation=self.interpolation)
        if c == 1: outimg = outimg[:,:,np.newaxis]
        return outimg

class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
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
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1+th, x1:x1+tw, ...]


class RandomScale(object):

    def __init__(self, interpolation=cv2.INTER_CUBIC):
        self.interpolation = interpolation

    def __call__(self, img):
        # random_scale = random.sample([0.25, 0.5, 1.0], 1)
        random_scale = [1.0]
        w, h = img.shape[1], img.shape[0]
        w = int(w * random_scale[0])
        h = int(h * random_scale[0])

        return cv2.resize(img, dsize=(w, h),
                          interpolation=self.interpolation)


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, 1).reshape(img.shape)
            mask = cv2.flip(img, 1).reshape(mask.shape)
        return img, mask

class RandomVerticalFlip(object):
    """Randomly vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, 0).reshape(img.shape)
            mask = cv2.flip(img, 0).reshape(mask.shape)
        return img, mask

class RandomTransposeFlip(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = cv2.flip(img, -1).reshape(img.shape)
            mask = cv2.flip(img, -1).reshape(mask.shape)
        return img, mask

class RandomBlur(object):
    def __call__(self, img, mask):
        if random.random() < 0.8:
            # kernel_size = random.randrange(1, 19 + 1, 2)
            kernel_size = 19
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        return img, mask


class Convert(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img, mask):
        if img.ndim < 3:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)

        dtype = torch.FloatTensor
        img = torch.from_numpy(img).type(dtype)/255.0

        return img
