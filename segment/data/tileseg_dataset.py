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
from base.data.base_dataset import BaseDataset, get_augment_seq, RandomCrop

ia.seed(1)


class TileSegDataset(BaseDataset):
    """
        Same as TilePheno except instead of outputing the class label of the tile it outputs its ground truth
    """

    def __init__(self, opt):
        super(TileSegDataset, self).__init__()
        self.opt = opt
        self.paths = []
        tiles_path = Path(self.opt.data_dir)/'data'/'tiles'
        wsi_paths = [path for path in tiles_path.iterdir() if path.is_dir()]  # one per wsi image the tiles were derived from
        paths = [path for path in chain(*(wsi_path.glob(self.opt.image_glob_pattern) for wsi_path in wsi_paths))]
        assert paths, "Cannot be empty"
        with open(self.opt.split_file, 'r') as split_json:
            self.split = json.load(split_json)
        self.opt.phase = self.opt.phase if self.opt.phase != 'val' else 'test'  # check on test set during training (TEMP)
        phase_split = set(self.split[self.opt.phase])  # ~O(1) __contains__ check through hash table
        id_len = len(phase_split.pop())  # checks length of id
        self.paths = sorted(path for path in paths if path.parent.name[:id_len] in phase_split)
        assert self.paths, "Filtered paths list cannot be empty"
        self.gt_paths = [Path(str(path).replace('_img_', '_mask_')) for path in self.paths]  # FIXME adapt to annotations
        self.randomcrop = RandomCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)

    def __len__(self):
        return len(self.paths)

    def name(self):
        return "TileSegDataset"

    def setup(self):
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--segment_lumen', action='store_true', help="Whether the dataset should output separate labels for the lumen")
        parser.add_argument('--split_file', type=str,
                            default='/well/rittscher/projects/TCGA_prostate/TCGA/data/CVsplits/split0.json',
                            help="File containing data division in train - test split")
        parser.add_argument('--image_glob_pattern', type=str, default='*_*.png',
                            help='Pattern used to find images in each WSI / region folder')
        parser.add_argument('--coords_pattern', type=str, default='\((\w\.\w{1,3}),(\w{1,6}),(\w{1,6}),(\w{1,6}),(\w{1,6})\)_img_(\w{1,6}),(\w{1,6})')
        parser.add_argument('--area_based_input', action='store_true', help="For compatibility with first experiment, if true coords of tiles are relative to area they were extracted from")
        return parser

    def rescale(self, image, resolution_data, gt=None):
        """
        Rescale to desired resolution, if tiles are at a different millimeter per pixel (mpp) scale
        mpp replaces fine_size to decide rescaling.
        Also, rescaling is done before cropping/padding, to ensure that final image is of desired size and resolution
        :param image:
        :param resolution_data:
        :param gt: optionally scale and pad / random crop ground truth as for the image
        :return:
        """
        if gt and (gt.ndim == 3 and gt.shape[2] == 3):
            gt = gt[..., 0]  # take only one channel of the 3 identical RGB values
        if gt and (not self.opt.segment_lumen):
            gt[gt > 0] = 255
        target_mpp, read_mpp = self.opt.mpp, resolution_data['read_mpp']
        if not np.isclose(target_mpp, read_mpp, rtol=0.01, atol=0.1):  # total tolerance = rtol*read_mpp + atol
            # if asymmetrical, crop image
            resize_factor = target_mpp / read_mpp
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
        return image, gt

    def get_area_coords_info(self, image_path):
        """
        Function used to extract coord info of tile WHEN the tile is assumed to come from a region of the slide, and its
        tile specific coords refer to this area rather than to the coords inside the WSI
        :param image_path:
        :return:
        """
        coords_info = re.search(self.opt.coords_pattern, image_path.name).groups()  # tuple with all matched groups
        downsample = float(coords_info[0])  # downsample is a float
        area_x, area_y, area_w, area_h, tile_x, tile_y = tuple(int(num) for num in coords_info[1:])
        coords_info = {'downsample': downsample,
                'area_x': area_x, 'area_y': area_y, 'area_w': area_w, 'area_h': area_h,
                'tile_x': tile_x, 'tile_y': tile_y}
        return coords_info

    def get_tile_coords_info(self, image_path):
        """
        Function used to extract coords when the coords in the tile name refer to a location in the WSI (upper left corner)
        :param image_path:
        :return:
        """
        coords_info = re.search(self.opt.coords_pattern, image_path.name).groups()  # tuple with all matched groups
        coords_info = {'tile_x': int(coords_info[0]), 'tile_y': int(coords_info[1])}
        return coords_info

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        resolution_data = json.load(open(image_path.parent / 'resolution.json', 'r'))
        image = imageio.imread(image_path)
        gt_path = self.gt_paths[idx]
        gt = imageio.imread(gt_path)
        image = self.rescale(image, resolution_data, gt=gt)
        if not (isinstance(gt, np.ndarray) and gt.ndim > 0):
            raise ValueError("{} is not valid".format(gt_path))
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
            bg_thresh, lumen_thresh = 30, 225
            gt[np.logical_and(gt < lumen_thresh, gt > bg_thresh)] = 1
            gt[gt >= bg_thresh] = 2
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
        # get coords info of tile wrt WSI
        if self.opt.area_based_input:
            coords_info = self.get_area_coords_info(image_path)
        else:
            coords_info = self.get_tile_coords_info(image_path)
        return dict(
            input=image,
            target=gt,
            input_path=str(image_path),
            target_path=str(gt_path),
            **coords_info,
        )


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


class TestDataset(TileSegDataset):
    def __init__(self, dir_, tile_size=256, bad_folds=[]):
        super(TestDataset, self).__init__()
        if bad_folds:
            for image_file, label_file in zip(self.paths, self.label):
                image_name = os.path.basename(image_file)
                label_name = os.path.basename(label_file)
                assert(image_name[0:10] == label_name[0:10])
                isbad = any([bad_fold in image_name for bad_fold in bad_folds])
                if isbad:
                    self.paths.remove(image_file)
                    self.label.remove(label_file)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(phase="test")
        return parser