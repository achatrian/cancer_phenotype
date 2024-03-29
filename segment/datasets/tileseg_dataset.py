import os
import glob
import re
import cv2
from pathlib import Path
from itertools import product, chain
import json
import imageio
import torch
import numpy as np
import imgaug as ia
import warnings
from base.datasets.base_dataset import BaseDataset, get_augment_seq, RandomCrop
ia.seed(1)


class TileSegDataset(BaseDataset):
    """
        Same as TilePheno except instead of outputing the class label of the tile it outputs its ground truth
    """

    def __init__(self, opt):
        super(TileSegDataset, self).__init__(opt)
        self.paths = []
        split_tiles_path = Path(self.opt.data_dir) / 'data' / 'cross_validate' / (
                    re.sub('.json', '', opt.split_file) + f'_tiles_{self.opt.phase}.txt')
        split_tiles_path = str(split_tiles_path)
        # read resolution data - requires global tcga_resolution.json file
        with open(Path(self.opt.data_dir) / 'data' / 'cross_validate' / 'tcga_resolution.json', 'r') as resolution_file:
            self.resolutions = json.load(resolution_file)
        try:
            with open(split_tiles_path, 'r') as split_tiles_file:
                self.paths = [Path(image_path) for image_path in json.load(split_tiles_file)]
            print(f"Loaded {len(self.paths)} tile paths for split {Path(self.opt.split_file).name}")
        except FileNotFoundError as err:
            raise ValueError("Given path does not correspond to any split file") from err
        assert self.paths, "Cannot be empty"
        if self.opt.load_ground_truth:
            self.gt_paths = [Path(str(path).replace('_img_', '_mask_')) for path in self.paths]  # FIXME adapt to annotations
        self.randomcrop = RandomCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)
        self.read_resolution_data = True  # used to skip looking for resolution file
        self.label_interval_map = {
            'epithelium': (31, 225),
            'lumen': (225, 250),
            'background': (0, 30)
        }
        self.label_value_map = {
            'epithelium': 200,
            'lumen': 250,
            'background': 0
        }

    def __len__(self):
        return len(self.paths)

    def name(self):
        return "TileSegDataset"

    def setup(self):
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--load_ground_truth', action='store_true', help="Whether to attemp to load ground truth masks for the tiles")
        parser.add_argument('--one_class', action='store_true', help="Whether the dataset should merge all the labels into a background vs objects problem")
        parser.add_argument('--split_file', type=str, default='', help="File containing data division in train - test split")
        parser.add_argument('--image_glob_pattern', type=str, default='*_*.png', help='Pattern used to find images in each WSI / region folder')
        parser.add_argument('--area_based_input', action='store_true', help="For compatibility with first experiment, if true coords of tiles are relative to area they were extracted from")
        parser.add_argument('--mpp', type=float, default=0.4, help="Resolution to read tiles at")
        return parser

    def rescale(self, image, resolution_data=None, gt=None):
        """
        Rescale to desired resolution, if tiles are at a different millimeter per pixel (mpp) scale
        mpp replaces fine_size to decide rescaling.
        Also, rescaling is done before cropping/padding, to ensure that final images is of desired size and resolution
        :param image:
        :param resolution_data:
        :param gt: optionally scale and pad / random crop ground truth as for the images
        :return:
        """
        if gt and (gt.ndim == 3 and gt.shape[2] == 3):
            gt = gt[..., 0]  # take only one channel of the 3 identical RGB values
        if gt and self.opt.one_class:
            gt[gt > 0] = 255
        if resolution_data:
            target_mpp, read_mpp = self.opt.mpp, resolution_data['read_mpp']
            if not np.isclose(target_mpp, read_mpp, rtol=0.01, atol=0.1):  # total tolerance = rtol*read_mpp + atol
                # if asymmetrical, crop images
                resize_factor = read_mpp / target_mpp
                image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
                if gt:
                    gt = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

        if image.shape[0:2] != (self.opt.patch_size,) * 2:
            too_narrow = image.shape[1] < self.opt.patch_size
            too_short = image.shape[0] < self.opt.patch_size
            if too_narrow or too_short:
                # pad if needed
                delta_w = self.opt.patch_size - image.shape[1] if too_narrow else 0
                delta_h = self.opt.patch_size - image.shape[0] if too_short else 0
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
                if gt:
                    gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_REFLECT)
            if image.shape[0] > self.opt.patch_size or image.shape[1] > self.opt.patch_size:
                if gt:
                    cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
                    cat = self.randomcrop(cat)
                    image = cat[:, :, 0:3]
                    gt = cat[:, :, 3]
                else:
                    image = self.randomcrop(image)

        if not resolution_data:
            # scale images using fine_size -- DEPRECATED --
            sizes = (self.opt.fine_size,) * 2
            image = cv2.resize(image, sizes, interpolation=cv2.INTER_AREA)
            if gt:
                gt = cv2.resize(gt, sizes, interpolation=cv2.INTER_AREA)

        return image, gt

    def get_area_coords_info(self, image_path):
        """
        Function used to extract coord info of tile WHEN the tile is assumed to come from a region of the slide, and its
        tile specific coords refer to this area rather than to the coords inside the WSI
        :param image_path:
        :return:
        """
        coords_info = re.search(
            '\((\w\.\w{1,3}),(\w{1,6}),(\w{1,6}),(\w{1,6}),(\w{1,6})\)_img_(\w{1,6}),(\w{1,6})',
            image_path.name).groups()  # tuple with all matched groups
        downsample = float(coords_info[0])  # downsample is a float
        area_x, area_y, area_w, area_h, tile_x, tile_y = tuple(int(num) for num in coords_info[1:])
        coords_info = {'downsample': downsample,
                'area_x': area_x, 'area_y': area_y, 'area_w': area_w, 'area_h': area_h,
                'tile_x': tile_x, 'tile_y': tile_y}
        coords_info['x_offset'] = area_x + tile_x
        coords_info['y_offset'] = area_y + tile_y
        coords_info['slide_id'] = re.match('.+?(?=_TissueTrain_)', Path(image_path).name).group()  # tested on regex101.com
        return coords_info

    def get_tile_coords_info(self, image_path):
        """
        Function used to extract coords when the coords in the tile name refer to a location in the WSI (upper left corner)
        :param image_path:
        :return:
        """
        coords_info = re.search(self.opt.coords_pattern, image_path.name).groups()  # tuple with all matched groups
        coords_info = {'x_offset': int(coords_info[0]), 'y_offset': int(coords_info[1])}
        return coords_info

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        if self.read_resolution_data:
            try:
                resolution_data = json.load(open(image_path.parent / 'resolution.json', 'r'))
            except FileNotFoundError:
                self.read_resolution_data = False
                resolution_data = None
        else:
            resolution_data = None
        image = imageio.imread(image_path)
        if self.opt.load_ground_truth:  # FIXME are all these options really needed ? Code is hard to read
            # process images and ground truth together to keep spatial correspondence
            gt_path = self.gt_paths[idx]
            gt = imageio.imread(gt_path)
            image, gt = self.rescale(image, resolution_data, gt=gt)
            if not (isinstance(gt, np.ndarray) and gt.ndim > 0):
                raise ValueError("{} is not valid".format(gt_path))
            # im aug
            image, gt = self.augment_image(image, gt)
            if not self.opt.segment_lumen:
                gt[gt < 255] = 0
                gt[gt != 0] = 1
            else:
                bg_thresh, lumen_thresh = self.label_interval_map['background'][1], self.label_interval_map['lumen'][1]
                gt[np.logical_and(gt < lumen_thresh, gt > bg_thresh)] = 1
                gt[gt >= bg_thresh] = 2
                gt[np.logical_and(gt != 1, gt != 2)] = 0
            assert (len(gt.shape) == 2)
            gt = torch.from_numpy(gt.copy()).long()  # change to FloatTensor for BCE
        else:
            image, _ = self.rescale(image, resolution_data)  # gt is none here
        # scale between 0 and 1
        image = image/255.0
        # normalised images between -1 and 1
        image = (image - 0.5)/0.5
        # convert to torch tensor
        if image.shape[2] > 3:
            image = image[..., :3]
        assert(image.shape[2] == 3)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        # get coords info of tile wrt WSI
        if self.opt.area_based_input:
            coords_info = self.get_area_coords_info(image_path)
        else:
            coords_info = self.get_tile_coords_info(image_path)
        data = dict(
            input=image,
            input_path=str(image_path),
            **coords_info,
        )
        if self.opt.load_ground_truth:
            data['target'] = gt
            data['target_path'] = str(gt_path)
        return data

    def make_subset(self, selector='', selector_type='match', store_name='paths'):
        # TODO to test
        if hasattr(self.opt, 'slide_id'):
            slide_ids = self.split['train'] if self.opt.is_train else self.split['test']
            if not any(slide_id in self.opt.slide_id for slide_id in slide_ids):
                raise ValueError(
                    f"Slide not in {'train' if self.opt.is_train else 'test'} split for {self.opt.split_file}")
        super().make_subset(self.opt.slide_id)
        if self.opt.load_ground_truth:
            super().make_subset(self.opt.slide_id, store_name='gt_paths')


####

class AugDataset(TileSegDataset):

    def __init__(self, dir_, aug_dir, mode, tile_size, augment=0, generated_only=False):
        super(AugDataset, self).__init__()
        if generated_only:
            self.paths = []
            self.label = []
        n = "[0-9]"
        names = ["_rec1_", "_rec2_"] #, "_gen1_", "_gen2_"]
        no_aug_len = len(self.paths)
        paths = []
        for gl_idx, x, y, name in product(range(1, 7), range(1, 6), range(1, 6), names):
            to_glob = os.path.join(aug_dir, 'gland_img_' + n * gl_idx + '_(' + n * x + ',' + n * y + ')' + \
                                   name + 'fake_B.png')
            paths += glob.glob(to_glob)
        self.paths += paths
        self.label += [x.replace('fake_B', 'real_A') for x in paths]
        assert (len(self.paths) > no_aug_len)