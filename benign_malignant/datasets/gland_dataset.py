from pathlib import Path
from PIL import Image
import re
import json
import numpy as np
import cv2
import imageio
import torch
from base.datasets.base_dataset import BaseDataset, get_augment_seq, RandomCrop


class GlandDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
        if self.opt.is_train:
            raise ValueError("GlandDataset should only be used in apply mode")
        self.opt.data_dir = Path(self.opt.data_dir)
        self.labels, self.ground_truth_paths = [], []
        # read in benign and malignant tile paths
        self.ground_truth_paths = []
        self.tiles_dir = self.opt.data_dir/'data'/'tiles'/'epithelium'
        for slide_dir in self.tiles_dir.iterdir():
            if not slide_dir.is_dir():
                continue
            images_paths = list(path for path in slide_dir.iterdir() if 'image' in path.name)
            self.paths.extend(images_paths)
            self.ground_truth_paths.extend(path.parent/path.name.replace('image', 'mask') for path in images_paths)
        assert len(self.paths) > 0, "Paths can't be empty"
        try:  # assume all tiles were acquired at same mpp
            tiles_info_path = next((self.tiles_dir/'logs').iterdir())
            with open(tiles_info_path, 'r') as tiles_info_file:
                self.tiles_info = json.load(tiles_info_file)  # read resolution of tiles
        except StopIteration:
            self.tiles_info = None

    def name(self):
        return "GlandDataset"

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--mpp', type=float, default=0.25, help="Resolution to load the slides at")
        parser.add_argument('--load_ground_truth', action='store_true',
                            help="Whether not to load the focus area")
        return parser

    def rescale(self, image, dataset_mpp=0.25, ground_truth=None):
        r"""
        Rescale to desired resolution, if tiles are at a different millimeter per pixel (mpp) scale
        mpp replaces fine_size to decide rescaling.
        Also, rescaling is done before cropping/padding, to ensure that final images is of desired size and resolution
        :param image:
        :param dataset_mpp:
        :param ground_truth: optionally scale and pad / random crop ground truth as for the images
        :return:
        """
        if ground_truth is not None and (ground_truth.ndim == 3 and ground_truth.shape[2] == 3):
            ground_truth = ground_truth[..., 0]  # take only one channel of the 3 identical RGB values
        target_mpp, read_mpp = self.opt.mpp, dataset_mpp
        if not np.isclose(target_mpp, read_mpp, rtol=0.01, atol=0.1):  # total tolerance = rtol*read_mpp + atol
            # if asymmetrical, crop images
            resize_factor = read_mpp / target_mpp
            image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            if ground_truth:
                ground_truth = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
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
                if ground_truth:
                    ground_truth = cv2.copyMakeBorder(ground_truth, top, bottom, left, right, cv2.BORDER_REFLECT)
            if image.shape[0] > self.opt.patch_size or image.shape[1] > self.opt.patch_size:
                if ground_truth is not None:
                    cat = np.concatenate([image, ground_truth[:, :, np.newaxis]], axis=2)
                    cat = self.randomcrop(cat)
                    image = cat[:, :, 0:3]
                    ground_truth = cat[:, :, 3]
                else:
                    image = self.randomcrop(image)
        return image, ground_truth

    def __getitem__(self, item):
        image_path = self.paths[item]
        image = imageio.imread(image_path)
        ground_truth_path = self.ground_truth_paths[item]
        ground_truth = imageio.imread(ground_truth_path) if self.opt.load_ground_truth else None
        if image.shape[-1] == 4:  # convert RGBA to RGB
            image = np.array(Image.fromarray(image.astype('uint8'), 'RGBA').convert('RGB'))
        # process images and ground truth together to keep spatial correspondence
        if self.opt.load_ground_truth:
            assert ground_truth.ndim == 2, "Check ground_truth format"
            if not (isinstance(ground_truth, np.ndarray) and ground_truth.ndim > 0):
                raise ValueError("{} is not valid".format(ground_truth_path))
        image, ground_truth = self.rescale(image, ground_truth=ground_truth,
                                           dataset_mpp=self.opt.set_mpp)
        # im aug
        if self.opt.augment_level:
            seq_det = self.aug_seq.to_deterministic()  # needs to be called for every batch https://github.com/aleju/imgaug
            image = seq_det.augment_image(image)
            if self.opt.load_ground_truth:
                ground_truth = np.squeeze(
                    seq_det.augment_image(np.tile(ground_truth[..., np.newaxis], (1, 1, 3)), ground_truth=True))
                ground_truth = ground_truth[..., 0]
        if self.opt.load_ground_truth:
            ground_truth = torch.from_numpy(ground_truth.copy()).long()  # change to FloatTensor for BCE
        # scale between 0 and 1
        image = image / 255.0
        # normalised images between -1 and 1
        image = (image - 0.5) / 0.5
        # convert to torch tensor
        assert (image.shape[-1] == 3)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        # coordinates offset
        groups = re.match(r'epithelium_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)?', Path(image_path).name).groups()
        x_offset, y_offset, w, h, tile_num = groups
        data = dict(
            input=image,
            input_path=str(image_path),
            x_offset=int(x_offset),
            y_offset=int(y_offset),
            width=int(w),
            height=int(h),
            tile_num=int(tile_num) if tile_num is not None else 0
        )
        if self.opt.load_ground_truth:
            data.update(
                ground_truth=ground_truth,
                ground_truth_path=str(ground_truth_path)
            )
        return data



