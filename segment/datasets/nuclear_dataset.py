import glob
import os
import warnings
import cv2
import torch
import numpy as np
import imgaug as ia
from base.datasets.base_dataset import BaseDataset, get_augment_seq, RandomCrop

ia.seed(1)


class NuclearDataset(BaseDataset):

    def __init__(self, opt):
        super(NuclearDataset, self).__init__(opt)
        self.image_files = []
        self.label = []

        phase_dir = os.path.join(self.opt.data_dir, self.opt.phase)
        folders = [name for name in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, name))]

        image_files, label_files = [], []
        for fold in folders:
            image_files += glob.glob(os.path.join(phase_dir, fold, 'images', '*.png'))
            label_files += glob.glob(os.path.join(phase_dir, fold, 'masks_001', '*_gt.png'))

        self.image_files = image_files
        self.label_files = label_files
        assert self.image_files
        assert self.label_files
        assert len(self.image_files) == len(self.label_files)

        self.randomcrop = RandomCrop(self.opt.patch_size)

        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)

    def __len__(self):
        return len(self.image_files)

    def name(self):
        return "NuclearDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(data_dir="/gpfs0/well/rittscher/users/achatrian/ProstateCancer/Dataset/03_nucleus")
        return parser

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

        if image.shape[0:2] != (self.opt.patch_size,)*2:
            too_narrow = image.shape[1] < self.opt.patch_size
            too_short = image.shape[0] < self.opt.patch_size
            if too_narrow or too_short:
                delta_w = self.opt.patch_size - image.shape[1] if too_narrow else 0
                delta_h = self.opt.patch_size - image.shape[0] if too_short else 0
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
                gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_REFLECT)

            if image.shape[0] > self.opt.patch_size or image.shape[1] > self.opt.patch_size:
                cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
                cat = self.randomcrop(cat)
                image = cat[:, :, 0:3]
                gt = cat[:, :, 3]

        if self.opt.patch_size > self.opt.fine_size:
            # scale images
            sizes = (self.opt.fine_size, ) * 2
            image = cv2.resize(image, sizes, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, sizes, interpolation=cv2.INTER_AREA)

        # im aug
        cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
        if self.opt.augment_level:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # but future extensions could be causing problems
                cat = self.aug_seq.augment_image(cat)
        image = cat[:, :, 0:3]
        gt = cat[:, :, 3]
        gt[gt < 255] = 0
        gt[gt != 0] = 1

        # scale between 0 and 1
        image = image/255.0
        # normalised images between -1 and 1
        image = (image - 0.5)/0.5

        # convert to torch tensor
        assert(image.shape[-1] == 3)
        assert(len(gt.shape) == 2)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        gt = torch.from_numpy(gt.copy()).long()  # change to FloatTensor for BCE
        return {'input': image, 'target': gt, 'input_path': img_name, 'target_path': gt_name}
