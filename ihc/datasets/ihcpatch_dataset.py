from pathlib import Path
import pandas as pd
import json
import random
from datetime import datetime
import numpy as np
import cv2
import torch
import imageio
from PIL import Image
from base.datasets.base_dataset import BaseDataset, get_augment_seq, RandomCrop


class IHCPatchDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.data_dir = Path(self.opt.data_dir)
        # read file describing what type of stains the images are
        self.slide_data = pd.read_csv(self.opt.ihc_data_file)
        # assume data organization is dd/data/tiles/Focus#/slide_id
        self.tiles_dir = self.opt.data_dir/'data'/self.opt.tiles_dirname
        self.all_paths = []  # store both H&E and CK5 images
        for focus_path in self.tiles_dir.iterdir():
            if not focus_path.is_dir():
                continue
            for slide_path in focus_path.iterdir():
                self.all_paths += list(slide_path.glob('*_image.png'))
        if not self.all_paths:
            raise ValueError(f"No image tiles in {self.opt.data_dir}")
        self.all_focus_areas_paths = [path.parent / path.name.replace('image', 'mask') for path in self.all_paths]
        self.all_focus_areas_paths = [path for path in self.all_focus_areas_paths if path.exists()]
        # separate tile paths of H&E slides and CK5 slides into different containers
        self.paths, self.ck5_paths, self.focus_areas_paths, self.ck5_focus_areas_paths = [], [], [], []
        self.labels, self.ck5_labels = [], []  # ambiguous (IHC request) = 1, unambiguous (No IHC request) 0
        for i, slide_row in self.slide_data.iterrows():
            slide_id, staining_type, case_type = slide_row['Image'], slide_row['Staining code'], slide_row['Case type']
            if staining_type is np.nan:
                slide_paths = [path for path in self.all_paths if path.parent.name == slide_id
                               and path.name.endswith('_image.png')]
                self.paths += slide_paths
                assert case_type in {'Real', 'Control'}, "Unrecognized case type"
                self.labels += [1 if case_type == 'Real' else 0] * len(slide_paths)
                self.focus_areas_paths += [path for path in self.all_focus_areas_paths if path.parent.name == slide_id
                                           and path.name.endswith('_mask.png')]
            elif staining_type == 'CK5':
                slide_paths = [path for path in self.all_paths if path.parent.name == slide_id
                               and path.name.endswith('_image.png')]
                self.ck5_paths += slide_paths
                self.ck5_labels += [1 if case_type == 'Real' else 0] * len(slide_paths)
                self.ck5_focus_areas_paths += [path for path in self.all_focus_areas_paths if
                                               path.parent.name == slide_id
                                               and path.name.endswith('_mask.png')]
            elif staining_type in {'panCK', 'Other', 'Staining code', '34BE12'}:
                continue  # TODO decide how to deal with panCK slides and other stains
            else:
                raise ValueError(f"Unknown staining type: {staining_type}")
        assert self.paths and len(self.paths) == len(self.focus_areas_paths) and \
               len(self.ck5_paths) == len(self.ck5_focus_areas_paths), "Focus tile mismatch"
        print(f"Found {len(self.paths)} H&E tiles ({sum(self.labels)} IHC, {len(self.labels) - sum(self.labels)} Controls) and {len(self.ck5_paths)} CK5 tiles ({sum(self.ck5_labels)} IHC, {len(self.ck5_labels) - sum(self.ck5_labels)} Controls)")
        self.make_train_test_split()  # only keeps paths for specified phase
        if self.opt.phase == 'train':
            self.paths, self.labels, self.ck5_paths, self.ck5_labels = self.train_test_split['train_paths'], self.train_test_split['train_labels'], self.train_test_split['train_ck5_paths'], self.train_test_split['train_ck5_labels']
        elif self.opt.phase in {'val', 'test'}:
            self.paths, self.labels, self.ck5_paths, self.ck5_labels = self.train_test_split['test_paths'], self.train_test_split['test_labels'], self.train_test_split['test_ck5_paths'], self.train_test_split['test_ck5_labels']
        assert len(self.paths) == len(self.labels) and len(self.ck5_paths) == len(self.ck5_labels),  "Same # of paths and labels (shouldn't be possible unless file is tampered with)"
        print(f"Selected data for {self.opt.phase} split.")
        self.randomcrop = RandomCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)
        # read resolution info on tiles
        try:
            tiles_info_path = next(self.tiles_dir.glob('tiles_info*'))
            with open(tiles_info_path, 'r') as tiles_info_file:
                self.tiles_info = json.load(tiles_info_file)  # read resolution of tiles
        except StopIteration:
            self.tiles_info = None
        # TODO divide into train-test splits, save split info and keep only train / test depending on opt.phase
        # TODO check that images are all at same resolution ? (Maybe not here though)
        # TODO only applying annotations to CK5 images for a lot of cases -- apply to all!!

    def name(self):
        return 'IHCPatchDataset'

    def __len__(self):
        return len(self.paths) if self.opt.stain == 'HE' else len(self.ck5_paths)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--ihc_data_file', type=Path,
                            default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_12_12_19.csv')
        parser.add_argument('--tiles_dirname', type=str, default='tiles', help="Name of folder containing the loaded tiles")
        parser.add_argument('--stain', type=str, default='HE', choices=['HE', 'CK5'], help="What stain data to use")
        parser.add_argument('--train_fraction', type=float, default=0.6, help="Fraction of samples used for training")
        parser.add_argument('--no_ground_truth', action='store_true',
                            help="Whether not to load the focus area")  # TODO this needed?
        parser.add_argument('--mpp', type=float, default=0.25, help="Resolution to load the slides at")
        parser.add_argument('--overwrite_split', action='store_true', help="Write split file")
        return parser

    def get_sampler(self):
        if self.opt.stain == 'HE':
            weight = sum(self.labels)/len(self.labels)
            weights = [1/weight if label else 1 for label in self.labels]
        elif self.opt.stain == 'CK5':
            weight = sum(self.ck5_labels)/len(self.ck5_labels)
            weights = [1/weight if label else 1 for label in self.labels]
        else:
            raise ValueError(f"Invalid stain type {self.opt.stain}")
        return torch.utils.data.WeightedRandomSampler(weights, len(self))

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

    def __getitem__(self, idx):
        if self.opt.stain == 'HE':
            image_path, ground_truth_path = self.paths[idx], self.focus_areas_paths[
                idx] if not self.opt.no_ground_truth else None
        elif self.opt.stain == 'CK5':
            image_path, ground_truth_path = self.ck5_paths[idx], self.ck5_focus_areas_paths[
                idx] if not self.opt.no_ground_truth else None
        else:
            raise ValueError(f"Unrecognized stain type {self.opt.stain}")
        image = imageio.imread(image_path)
        ground_truth = imageio.imread(ground_truth_path) if not self.opt.no_ground_truth else None
        if image.shape[-1] == 4:  # convert RGBA to RGB
            image = np.array(Image.fromarray(image.astype('uint8'), 'RGBA').convert('RGB'))
        # process images and ground truth together to keep spatial correspondence
        if not self.opt.no_ground_truth:
            assert ground_truth.ndim == 2, "Check ground_truth format"
            if not (isinstance(ground_truth, np.ndarray) and ground_truth.ndim > 0):
                raise ValueError("{} is not valid".format(ground_truth_path))
        image, ground_truth = self.rescale(image, ground_truth=ground_truth,
                                           dataset_mpp=self.tiles_info['mpp'] if self.tiles_info is not None else 0.25)
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
        data = dict(
            input=image,
            input_path=str(image_path),
            target=self.labels[idx] if self.opt.stain == 'HE' else self.ck5_labels[idx],
        )
        if not self.opt.no_ground_truth:
            data.update(
                ground_truth=ground_truth,
                ground_truth_path=str(ground_truth_path)
            )
        return data

    def make_train_test_split(self):
        from sklearn.model_selection import train_test_split
        assert self.opt.train_fraction <= 0.99
        # split file is in the experiment folder
        split_path = Path(self.opt.checkpoints_dir)/self.opt.experiment_name/'train_test_split.json'
        try:
            if self.opt.overwrite_split:
                raise FileNotFoundError("Overwrite")
            with split_path.open('r') as split_file:
                self.train_test_split = json.load(split_file)
            self.train_test_split['train_paths'] = [Path(s) for s in self.train_test_split['train_paths']]
            self.train_test_split['test_paths'] = [Path(s) for s in self.train_test_split['test_paths']]
            self.train_test_split['train_ck5_paths'] = [Path(s) for s in self.train_test_split['train_ck5_paths']]
            self.train_test_split['test_ck5_paths'] = [Path(s) for s in self.train_test_split['test_ck5_paths']]
            if self.train_test_split['train_fraction'] != self.opt.train_fraction:
                raise FileNotFoundError('Train fraction is different from the required value')
            if not set(self.train_test_split['train_paths']) <= set(self.paths) or \
                    not set(self.train_test_split['test_paths']) <= set(self.paths) or \
                    not set(self.train_test_split['train_ck5_paths']) <= set(self.ck5_paths) or \
                    not set(self.train_test_split['test_ck5_paths']) <= set(self.ck5_paths):
                raise ValueError(f"Mismatch between read paths and paths in split file: {split_path}")
        except (FileNotFoundError, json.JSONDecodeError):
            train_paths, test_paths, train_labels, test_labels = train_test_split(self.paths, self.labels,
                                                                                  train_size=self.opt.train_fraction)
            train_ck5_paths, test_ck5_paths, train_ck5_labels, test_ck5_labels = train_test_split(self.ck5_paths,
                                                                                                  self.ck5_labels,
                                                                                                  train_size=self.opt.train_fraction)
            self.train_test_split = {
                'train_paths': [str(path) for path in train_paths],
                'test_paths': [str(path) for path in test_paths],
                'train_labels': train_labels,
                'test_labels': test_labels,
                'train_ck5_paths': [str(path) for path in train_ck5_paths],
                'test_ck5_paths': [str(path) for path in test_ck5_paths],
                'train_ck5_labels': train_ck5_labels,
                'test_ck5_labels': test_ck5_labels,
                'train_fraction': self.opt.train_fraction,
                'date': str(datetime.now())[:10]
            }
            with open(split_path, 'w') as split_file:
                json.dump(self.train_test_split, split_file)

