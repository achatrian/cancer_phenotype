from pathlib import Path
from imageio import imread
import numpy as np
from cv2 import copyMakeBorder, BORDER_REFLECT
import torch
from base.datasets.base_dataset import BaseDataset, get_augment_seq
from base.utils.utils import bytes2human


class TissueDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__(opt)
        thumbnails_dir = Path(self.opt.data_dir, 'data', 'thumbnails')
        self.paths = sorted(list(thumbnails_dir.glob(f'thumbnail_*')), key=lambda p: p.name)
        self.masks_paths = sorted(list(thumbnails_dir.glob(f'mask_*')), key=lambda p: p.name)
        self.slides_ids = set(path.name.split('thumbnail_')[1] for path in self.paths)
        if len(self.paths) != len(self.masks_paths):
            different_ids = set(path.name.split('mask_')[1] for path in self.masks_paths)
            thumbnails_with_mask = self.slides_ids - self.slides_ids ^ different_ids
            self.paths = [path for path in self.paths if path.name.split('thumbnail_')[1] in thumbnails_with_mask]
        assert len(self.paths) == len(self.masks_paths), "The data folder must contain a mask for each thumbnail"
        self.expected_shape = (self.opt.patch_size, self.opt.patch_size, 3)
        if self.opt.max_image_size is not None:
            removed = []
            for i, image_path in reversed(tuple(enumerate(self.paths))):
                if image_path.stat().st_size > self.opt.max_image_size:
                    removed.append(self.paths.pop(i))
                    self.masks_paths.pop(i)
            print(f"{len(removed)} images exceeded the set max image size of {bytes2human(self.opt.max_image_size)}")

    def name(self):
        return "TissueDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--max_image_size', type=int, default=None, help="Max size of image ")
        return parser

    def __getitem__(self, item):
        image, mask = imread(self.paths[item]), imread(self.masks_paths[item])
        if self.opt.is_train:
            delta_w, delta_h = self.opt.patch_size - image.shape[1], self.opt.patch_size - image.shape[0]
            if delta_h > 0 or delta_w > 0:
                delta_w_, delta_h_ = max(delta_w, 0), max(delta_h, 0)
                top, bottom = delta_h_ // 2, delta_h_ - (delta_h_ // 2)
                left, right = delta_w_ // 2, delta_w_ - (delta_w_ // 2)
                image = copyMakeBorder(image, top, bottom, left, right, BORDER_REFLECT)
                mask = copyMakeBorder(mask, top, bottom, left, right, BORDER_REFLECT)
            if delta_h < 0 or delta_w < 0:
                image = image[:self.opt.patch_size, :self.opt.patch_size]
                mask = mask[:self.opt.patch_size, :self.opt.patch_size]
        # masks were created with a bias of 100
        mask = mask - 100  # should fix the thumbnail thresholding script
        mask[mask > 0] = 1
        image, mask = self.augment_image(image, mask)
        # for old images with red background, whiten the background
        image = image/255.0
        # normalised images between -1 and 1
        image = (image - 0.5)/0.5
        assert 1 <= len(set(mask.flatten().tolist())) <= 2, f"loaded image is a binary tissue mask ({self.masks_paths[item]})"
        # convert to torch tensor
        if self.opt.is_train:
            assert tuple(image.shape) == self.expected_shape, f"image shape must match specification {self.expected_shape} != {tuple(image.shape)}"
        image = image[..., :3]  # remove alpha channel in case it's present
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()
        return dict(
            input=image,
            target=mask,
            input_path=str(self.paths[item]),
            target_path=str(self.masks_paths[item])
        )