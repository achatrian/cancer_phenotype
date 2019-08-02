from pathlib import Path
import json
import warnings
import numpy as np
import imageio
import cv2
import torch
from torchvision.transforms import ToTensor
from base.data.base_dataset import BaseDataset, RandomCrop, get_augment_seq


class InstanceDataset(BaseDataset):
    r"""Dataset to lod component instance tiles and associated ground truth masks"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.slides_paths = [path for path in Path(self.opt.data_dir).iterdir() if path.suffix in ('.svs', '.ndpi')]
        self.instance_tiles_path = Path(self.opt.data_dir)/'data'/'tiles'/self.opt.tissue_label
        # /data/tiles/tissue_label/slide_id/images/*
        self.paths = tuple(path for path in self.instance_tiles_path.glob(f'*/images/{self.opt.tissue_label}.png'))
        assert self.paths, "Paths must not be empty"
        self.masks_paths = tuple(Path(str(path).replace('images', 'masks')) for path in self.paths)
        self.to_tensor = ToTensor()
        self.randomcrop = RandomCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)
        self.tiles_info_paths = tuple(path for path in (self.instance_tiles_path/'logs').iterdir() if path.suffix == '.json')
        self.tiles_infos = []
        for tiles_info_path in self.tiles_info_paths:
            with open(tiles_info_path, 'r') as tiles_info_file:
                self.tiles_infos.append(json.load(tiles_info_file))
        self.label_interval_map = {  # FIXME not general
            'epithelium': (31, 225),
            'lumen': (225, 250),
            'background': (0, 30)
        }
        self.label_value_map = {
            'epithelium': 200,
            'lumen': 250,
            'background': 0
        }

    def name(self):
        return "InstanceDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--tissue_label', type=str, default='epithelium', help="Name of subdir in tiles dir where patches are read from")
        parser.add_argument('--mpp', type=float, default=0.5, help="loading resolution of image patches")
        return parser

    def __len__(self):
        return len(self.paths)

    def rescale(self, image, read_mpp, mask=None):
        """
        Rescale to desired resolution, if tiles are at a different millimeter per pixel (mpp) scale
        mpp replaces fine_size to decide rescaling.
        Also, rescaling is done before cropping/padding, to ensure that final image is of desired size and resolution
        :param image:
        :param read_mpp: mpp resolution of loaded patches
        :param mask: optionally scale and pad / random crop ground truth as for the image
        :return:
        """
        if mask and (mask.ndim == 3 and mask.shape[2] == 3):
            mask = mask[..., 0]  # take only one channel of the 3 identical RGB values
        if mask and self.opt.one_class:
            mask[mask > 0] = 255
        target_mpp, read_mpp = self.opt.mpp, read_mpp
        if not np.isclose(target_mpp, read_mpp, rtol=0.01, atol=0.1):  # total tolerance = rtol*read_mpp + atol
            # if asymmetrical, crop image
            resize_factor = target_mpp / read_mpp
            image = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            if mask:
                mask = cv2.resize(image, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

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
                if mask:
                    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_REFLECT)
            if image.shape[0] > self.opt.patch_size or image.shape[1] > self.opt.patch_size:
                if mask:
                    cat = np.concatenate([image, mask[:, :, np.newaxis]], axis=2)
                    cat = self.randomcrop(cat)
                    image = cat[:, :, 0:3]
                    mask = cat[:, :, 3]
                else:
                    image = self.randomcrop(image)

        return image, mask

    def __getitem__(self, item):
        image_path = self.paths[item]
        ground_truth_path = self.masks_paths[item]
        slide_id = image_path.parents[1]
        tiles_info = next(tiles_info for tiles_info in self.tiles_infos if tiles_info['slide_id'] == slide_id)
        image = imageio.imread(image_path)
        mask = imageio.imread(ground_truth_path)
        image, mask = self.rescale(image, read_mpp=tiles_info['mpp'], mask=mask)
        # augment image and mask
        if self.opt.augment_level:
            cat = np.concatenate([image, mask[:, :, np.newaxis]], axis=2)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # but future extensions could be causing problems
                cat = self.aug_seq.augment_image(cat)
            image = cat[:, :, 0:3]
            mask = cat[:, :, 3]
        # FIXME does not work with general classes
        bg_thresh, lumen_thresh = self.label_interval_map['background'][1], self.label_interval_map['lumen'][1]
        mask[np.logical_and(mask < lumen_thresh, mask > bg_thresh)] = 1
        mask[mask >= bg_thresh] = 2
        mask[np.logical_and(mask != 1, mask != 2)] = 0
        # scale between 0 and 1
        image = image / 255.0
        # normalised image between -1 and 1
        image = (image - 0.5) / 0.5
        # convert to torch tensor
        assert (image.shape[-1] == 3)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        return {
            'input': image,
            'input_path': image_path,
            'mask': mask,
            'mask_path': ground_truth_path
        }


        








