from pathlib import Path
import json
import re
import numpy as np
import imageio
import torch
from PIL.Image import Image

from ihc.datasets.ihcpatch_dataset import IHCPatchDataset


r"""
Additionally to training tiles, load all tiles classified as being ambiguous (i.e. part of a focus of interest) when they aren't.
"""


class IHCActiveDataset(IHCPatchDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt.no_ground_truth = True  # misclassified tiles don't have ground truth masks
        if self.opt.phase == 'train':
            misclassified_paths = list(self.opt.misclassified_dir.glob('*/*.png'))
            self.paths += misclassified_paths
            with open(self.opt.misclassified_dir/'misclassified_info.json', 'r') as misclassified_file:
                self.misclassified_info_file = json.load(misclassified_file)
            self.labels += [self.misclassified_info_file['examples_class']]*len(misclassified_paths)
            print(f"Adding {len(misclassified_paths)} misclassified tiles by experiment {self.opt.misclassified_dir.name} to dataset")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = IHCPatchDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--misclassified_dir', type=Path)
        return parser

    def name(self):
        return "IHCActiveDataset"

    def __getitem__(self, idx):
        try:
            if self.opt.stain == 'HE':
                image_path, ground_truth_path = self.paths[idx], self.focus_areas_paths[
                    idx] if not self.opt.no_ground_truth else None
            elif self.opt.stain == 'CK5':
                image_path, ground_truth_path = self.ck5_paths[idx], self.ck5_focus_areas_paths[
                    idx] if not self.opt.no_ground_truth else None
            else:
                raise ValueError(f"Unrecognized stain type {self.opt.stain}")
            image = imageio.imread(image_path)
        except ValueError as err:  #FIXME - HACK dataset is failing to read some tiles -- possible bad naming?
            print(err)
            print(self.paths[idx])
            if not self.opt.is_train:
                raise
            error = self.opt.is_train
            while error:
                try:
                    idx = int(np.random.randint(len(self)))
                    if self.opt.stain == 'HE':
                        image_path, ground_truth_path = self.paths[idx], self.focus_areas_paths[
                            idx] if not self.opt.no_ground_truth else None
                    elif self.opt.stain == 'CK5':
                        image_path, ground_truth_path = self.ck5_paths[idx], self.ck5_focus_areas_paths[
                            idx] if not self.opt.no_ground_truth else None
                    image = imageio.imread(image_path)
                    error = False
                except ValueError:
                    print(err)
                    print(self.paths[idx])
                    pass

        ground_truth = imageio.imread(ground_truth_path) if not self.opt.no_ground_truth else None
        if image.shape[-1] == 4:  # convert RGBA to RGB
            image = np.array(Image.fromarray(image.astype('uint8'), 'RGBA').convert('RGB'))
        # process images and ground truth together to keep spatial correspondence
        if not self.opt.no_ground_truth:
            assert ground_truth.ndim == 2, "Check ground_truth format"
            if not (isinstance(ground_truth, np.ndarray) and ground_truth.ndim > 0):
                raise ValueError("{} is not valid".format(ground_truth_path))
        image, ground_truth = self.rescale(image, ground_truth=ground_truth,
                                           dataset_mpp=self.tiles_info['mpp'])
        # im aug
        if self.opt.augment_level:
            seq_det = self.aug_seq.to_deterministic()  # needs to be called for every batch https://github.com/aleju/imgaug
            image = seq_det.augment_image(image)
            if not self.opt.no_ground_truth:
                ground_truth = np.squeeze(
                    seq_det.augment_image(np.tile(ground_truth[..., np.newaxis], (1, 1, 3)), ground_truth=True))
                ground_truth = ground_truth[..., 0]
        if not self.opt.no_ground_truth:
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
        try:
            groups = re.match(r'Focus\d_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)?', Path(image_path).name).groups()
            x_offset, y_offset, w, h, tile_num = groups
        except AttributeError:
            groups = re.match(r'(\d+)_(\d+)_(\d)_(\d.\d+)', Path(image_path).name).groups()
            x_offset, y_offset, classification, probability = groups
            w, h, tile_num = (0,)*3
        data = dict(
            input=image,
            input_path=str(image_path),
            target=self.labels[idx] if self.opt.stain == 'HE' else self.ck5_labels[idx],
            x_offset=int(x_offset),
            y_offset=int(y_offset),
            tile_num=int(tile_num) if tile_num is not None else 0
        )
        if not self.opt.no_ground_truth:
            data.update(
                ground_truth=ground_truth,
                ground_truth_path=str(ground_truth_path)
            )
        return data
