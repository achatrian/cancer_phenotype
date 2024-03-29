from pathlib import Path
import json
from collections import OrderedDict
import operator
import imageio
import torch
import numpy as np
import cv2
from skimage.color import rgba2rgb
import imgaug as ia
import warnings
from base.datasets.base_dataset import BaseDataset, get_augment_seq, RandomCrop
ia.seed(1)


class InstanceTilesDataset(BaseDataset):
    r"""
        Same as TilePheno except instead of outputing the class label of the tile it outputs its ground truth
    """

    def __init__(self, opt):
        super(InstanceTilesDataset, self).__init__(opt)
        self.paths = []
        self.opt.data_dir = Path(self.opt.data_dir)
        tiles_splits_path = self.opt.data_dir/'data'/'cross_validate'/'tiles_split.json'  # what if I had wanted to use a different layer?
        with open(tiles_splits_path, 'r') as tiles_splits_file:
            tiles_splits = json.load(tiles_splits_file)
        assert tiles_splits['n_splits'] > 0, "Nonzero number of splits"
        phase = 'test' if self.opt.phase == 'val' else self.opt.phase
        if self.opt.skip_labels is not None:
            with open(self.opt.skip_labels, 'r') as label_file:
                label_data = json.load(label_file)
            self.label_interval_map = label_data['interval_map']
            self.label_value_map = label_data['value_map']
        else:
            self.label_interval_map = {
                'background': (0, 30),
                'epithelium': (31, 225),
                'lumen': (225, 250)
            }
            self.label_value_map = {
                'background': 0,
                'epithelium': 200,
                'lumen': 250
            }
        self.label_interval_map = OrderedDict(sorted(self.label_interval_map.items(), key=operator.itemgetter(1)))
        self.label_value_map = OrderedDict(sorted(self.label_value_map.items(), key=operator.itemgetter(1)))
        slide_ids = tiles_splits[phase][self.opt.split_num]
        for layer in self.label_value_map:
            slides_tiles_dirs = tuple(self.opt.data_dir/'data'/'tiles'/layer/slide_id for slide_id in slide_ids)
            for slide_tiles_dir in slides_tiles_dirs:
                if slide_tiles_dir.is_dir():
                    self.paths.extend(path for path in slide_tiles_dir.iterdir() if path.name.endswith('_image.png'))
        with open(self.opt.data_dir/'data'/'tiles'/tiles_splits['roi_layer']/'logs'/(slides_tiles_dirs[0].name + '_tiles.json'), 'r') as tile_export_info_file:  # assume every slide has same info (?)
            self.resolution_data = json.load(tile_export_info_file)  # will be the same for all slides
        assert self.paths, "Cannot be empty"
        if not self.opt.no_ground_truth:
            self.gt_paths = [Path(str(path).replace('_image', '_mask')) for path in self.paths]
        self.randomcrop = RandomCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)
        self.last_crop = (0, 0)

    def __len__(self):
        return len(self.paths)

    def name(self):
        return "InstanceTilesDataset"

    def setup(self):
        pass

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--mpp', type=float, default=0.4, help="Resolution to read tiles at")
        parser.add_argument('--no_ground_truth', action='store_true', help="Whether to attemp to load ground truth masks for the tiles")
        parser.add_argument('--one_class', action='store_true', help="Whether the dataset should merge all the labels into a background vs objects problem")
        parser.add_argument('--split_num', type=int, default=0, help="Which split to use")
        parser.add_argument('--image_glob_pattern', type=str, default='*_*.png', help='Pattern used to find images in each WSI / region folder')
        parser.add_argument('--label_file', type=Path, default=None)
        return parser

    def rescale(self, image, gt=None):
        r"""
        Rescale to desired resolution, if tiles are at a different millimeter per pixel (mpp) scale
        mpp replaces fine_size to decide rescaling.
        Also, rescaling is done before cropping/padding, to ensure that final images is of desired size and resolution
        :param image:
        :param gt: optionally scale and pad / random crop ground truth as for the images
        :return:
        """
        origin = (0, 0)
        if gt is not None and (gt.ndim == 3 and gt.shape[2] == 3):
            gt = gt[..., 0]  # take only one channel of the 3 identical RGB values
        if gt is not None and self.opt.one_class:
            gt[gt > 0] = 255
        target_mpp, read_mpp = self.opt.mpp, self.resolution_data['mpp']
        if not np.isclose(target_mpp, read_mpp, rtol=0.01, atol=0.1):  # total tolerance = rtol*read_mpp + atol
            # if asymmetrical, crop images
            resize_factor = read_mpp / target_mpp
            image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            if gt is not None:
                gt = cv2.resize(gt, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

        if image.shape[0:2] != (self.opt.patch_size,) * 2:
            too_narrow = image.shape[1] < self.opt.patch_size
            too_short = image.shape[0] < self.opt.patch_size
            if too_narrow or too_short:
                delta_w = self.opt.patch_size - image.shape[1] if too_narrow else 0
                delta_h = self.opt.patch_size - image.shape[0] if too_short else 0
                if self.opt.phase == 'train':  # in the train phase contours extracted from reflected images will be wrong
                    # pad if needed
                    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                    left, right = delta_w // 2, delta_w - (delta_w // 2)
                    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
                    if gt is not None:
                        gt = cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_REFLECT)
                    origin = (top, bottom)
                else:
                    image = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT)
                    if gt is not None:
                        gt = cv2.copyMakeBorder(gt, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT)
            if image.shape[0] > self.opt.patch_size or image.shape[1] > self.opt.patch_size:
                if gt is not None:
                    cat = np.concatenate([image, gt[:, :, np.newaxis]], axis=2)
                    cat = self.randomcrop(cat)
                    origin = self.randomcrop.last_crop
                    image = cat[:, :, 0:3]
                    gt = cat[:, :, 3]
                else:
                    image = self.randomcrop(image)
                    origin = self.randomcrop.last_crop
        return image, gt, origin

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = imageio.imread(image_path)
        if image.shape[-1] == 4:  # assume this means rgba
            image = (rgba2rgb(image) * 255).astype(np.uint8)  # skimage funcs output image in RGB [0, 1] range
        assert image.ndim == 3 and image.shape[-1] == 3, "Check image format"
        if not self.opt.no_ground_truth:  # FIXME are all these options really needed ? Code is hard to read
            # process images and ground truth together to keep spatial correspondence
            gt_path = self.gt_paths[idx]
            gt = imageio.imread(gt_path)
            if gt.ndim == 3 and gt.shape[2] == 3:
                gt = gt[..., 0]
            assert gt.ndim == 2, "Check gt format"
            image, gt, origin = self.rescale(image, gt=gt)
            if not (isinstance(gt, np.ndarray) and gt.ndim > 0):
                raise ValueError("{} is not valid".format(gt_path))
            # im aug
            image, gt = self.augment_image(image, gt)
            # paint labels in
            for i, (label, interval) in enumerate(self.label_interval_map.items()):
                gt[np.logical_and(gt >= interval[0], gt <= interval[1])] = i
            gt[np.logical_and(gt < 0, gt > i)] = 0
            assert (len(gt.shape) == 2)
            gt = torch.from_numpy(gt.copy()).long()  # change to FloatTensor for BCE
        else:
            image, _, origin = self.rescale(image)  # gt is none here
        # scale between 0 and 1
        image = image/255.0
        # normalised images between -1 and 1
        image = (image - 0.5)/0.5
        # convert to torch tensor
        assert(image.shape[-1] == 3)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        # get coords info of tile wrt WSI
        coords = image_path.with_suffix('').name.split('_')
        data = dict(
            input=image,
            input_path=str(image_path),
            x_offset=int(coords[1]) + origin,  # different from roitiles
            y_offset=int(coords[2]) + origin
        )
        if not self.opt.no_ground_truth:
            data['target'] = gt
            data['target_path'] = str(gt_path)
        return data
