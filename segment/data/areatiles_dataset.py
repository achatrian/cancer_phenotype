import os
import glob
from itertools import product
import re
import cv2
import torch
import numpy as np
import imgaug as ia
import warnings
from base.data.base_dataset import BaseDataset, get_augment_seq, RandomCrop
ia.seed(1)


class AreaTilesDataset(BaseDataset):

    def __init__(self, opt):
        """
        Old dataset for loading gland images
        :param opt:
        """
        super(AreaTilesDataset, self).__init__()
        self.opt = opt
        self.file_list = []
        self.label = []
        phase_dir = os.path.join(self.opt.data_dir, self.opt.phase)
        folders = [name for name in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, name))]
        file_list = []
        for folder in folders:
            file_list += glob.glob(os.path.join(phase_dir, folder, 'tiles', '*_img_*.png'))
        if hasattr(self.opt, 'slide_id'):
            # If slide_id is given, retain only tiles for that slide
            for i, file_name in reversed(list(enumerate(file_list))):
                slide_id = re.match('.+?(?=_TissueTrain_)', os.path.basename(file_name)).group()  # tested on regex101.com
                if slide_id != self.opt.slide_id:
                    del file_list[i]
        self.file_list = file_list
        self.label = [x.replace('_img_', '_mask_') for x in file_list]
        self.randomcrop = RandomCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)

    def __len__(self):
        return len(self.file_list)

    def name(self):
        return "AreaTiles"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--segment_lumen', action='store_true')
        parser.add_argument('--coords_pattern', type=str, default='\((\w\.\w{1,3}),(\w{1,6}),(\w{1,6}),(\w{1,6}),(\w{1,6})\)_img_(\w{1,6}),(\w{1,6})')
        return parser

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
        if not self.opt.segment_lumen:
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

            # scale image
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
        if not self.opt.segment_lumen:
            gt[gt < 255] = 0
            gt[gt != 0] = 1
        else:
            gt[np.logical_and(gt < 180, gt > 30)] = 1
            gt[gt >= 180] = 2
            gt[np.logical_and(gt != 1, gt != 2)] = 0

        # scale between 0 and 1
        image = image/255.0
        # normalised image between -1 and 1
        image = (image - 0.5)/0.5

        # convert to torch tensor
        assert(image.shape[-1] == 3)
        assert(len(gt.shape) == 2)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        gt = torch.from_numpy(gt.copy()).long()  # change to FloatTensor for BCE

        coords_info = re.search(self.opt.coords_pattern, os.path.basename(img_name)).groups()  # tuple with all matched groups
        downsample = float(coords_info[0])  # downsample is a float
        area_x, area_y, area_w, area_h, tile_x, tile_y = tuple(int(num) for num in coords_info[1:])
        x_offset = area_x + tile_x
        y_offset = area_y + tile_y
        slide_id = re.match('.+?(?=_TissueTrain_)', os.path.basename(img_name)).group()  # tested on regex101.com

        return {'input': image, 'target': gt, 'input_path': img_name, 'target_path': gt_name,
                'slide_id': slide_id,
                'downsample': downsample,
                'x_offset': x_offset, 'y_offset': y_offset
                }


class AugDataset(AreaTilesDataset):

    def __init__(self, dir_, aug_dir, mode, tile_size, augment=0, generated_only=False):
        super(AugDataset, self).__init__()
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


class TestDataset(AreaTilesDataset):
    def __init__(self, dir_, tile_size=256, bad_folds=[]):
        super(TestDataset, self).__init__()
        if bad_folds:
            for image_file, label_file in zip(self.file_list, self.label):
                image_name = os.path.basename(image_file)
                label_name = os.path.basename(label_file)
                assert(image_name[0:10] == label_name[0:10])
                isbad = any([bad_fold in image_name for bad_fold in bad_folds])
                if isbad:
                    self.file_list.remove(image_file)
                    self.label.remove(label_file)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(phase="test")
        return parser
