from pathlib import Path
import json
import re
from imageio import imread
import numpy as np
import cv2
import torch
from scipy.stats import mode
from base.datasets.base_dataset import BaseDataset, RandomCrop, CenterCrop, get_augment_seq
from base.utils import debug


r"""Besides the Dataset to read challenge data, running this script takes the consensus of physician annotations"""


class GleasonChallengeDataset(BaseDataset):
    r""""""

    def __init__(self, opt):
        super().__init__(opt)
        self.paths = []
        self.opt.data_dir = Path(self.opt.data_dir)
        with open(self.opt.data_dir/'data'/'splits.json', 'r') as split_file:
            self.split = json.load(split_file)
        self.images_dir = self.opt.data_dir/'data'/'tiles'/'images'
        self.paths = sorted(self.images_dir.glob('*.png'), key=lambda p: p.name)
        # read annotations from all physicians
        # for i in range(1, 7):
        #     ground_truth_paths_ = sorted(Path(self.opt.data_dir, 'physician_annotations', f'Maps{i}_T').glob('*.png'))
        #     ground_truth_paths = []
        #     for j, path in enumerate(self.paths):
        #         try:
        #             ground_truth_path = next(ground_truth_path for ground_truth_path in ground_truth_paths_
        #                                      if ground_truth_path.name.startswith(path.name[:-4]))
        #             ground_truth_paths.append(ground_truth_path)
        #         except StopIteration:
        #             ground_truth_paths.append(None)  # fill with none if physician hasn't annotated path image
        #     assert len(ground_truth_paths) == len(self.paths), "all paths must be matched with a path or none"
        #     setattr(self, f'ground_truth_paths{i}', ground_truth_paths)
        # get consensus masks
        consensus_mask_dir = Path(self.opt.data_dir/'data'/'tiles'/'masks')
        try:
            self.consensus_mask_paths = sorted(consensus_mask_dir.iterdir(), key=lambda p: p.name)
        except FileNotFoundError:
            print("Consensus masks have not been created")
            raise
        if self.opt.phase == 'train':
            self.paths = [path for path in self.paths if any(path.name.startswith(fn[:-4])
                                                             for fn in self.split['train'])]
            self.consensus_mask_paths = [path for path in self.consensus_mask_paths if any(path.name.startswith(fn[:-4])
                                                             for fn in self.split['train'])]
        else:
            self.paths = [path for path in self.paths if any(path.name.startswith(fn[:-4])
                                                             for fn in self.split['test'])]
            self.consensus_mask_paths = []
        assert len(self.consensus_mask_paths) == len(self.paths), "all image path must correspond to a mask path"
        self.random_crop = RandomCrop(self.opt.patch_size)
        self.center_crop = CenterCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)

    def name(self):
        return "GleasonChallengeDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--zoom_factor', type=float, default=1.0)
        parser.set_defaults(patch_size=1024)
        return parser

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        mask_path = self.consensus_mask_paths[item] if self.opt.phase == 'train' else None
        image, mask = imread(path), imread(mask_path) if mask_path is not None else None
        origin = (0, 0)
        # rescale according to the zoom factor
        if self.opt.zoom_factor != 1.0:
            image = cv2.resize(image, None, fx=self.opt.zoom_factor, fy=self.opt.zoom_factor, interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, None, fx=self.opt.zoom_factor, fy=self.opt.zoom_factor, interpolation=cv2.INTER_NEAREST)
        # pad/crop if needed
        if image.shape[0:2] != (self.opt.patch_size,) * 2 and self.opt.phase == 'train':
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
                    if mask is not None:
                        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_REFLECT)
                    origin = (top, bottom)
                else:
                    image = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT)
                    if mask is not None:
                        mask = cv2.copyMakeBorder(mask, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT)
            if image.shape[0] >= self.opt.patch_size or image.shape[1] >= self.opt.patch_size:
                cat = np.concatenate((image, mask[..., np.newaxis]), axis=2) if mask is not None else image
                if self.opt.phase == 'train':
                    cat = self.random_crop(cat)
                    origin = self.random_crop.last_crop
                else:
                    cat = self.center_crop(cat)
                    origin = self.center_crop.last_crop
                image, mask = cat[..., :3], cat[..., 3].squeeze() if mask is not None else None
        assert image.shape[0:2] == (self.opt.patch_size,) * 2, "image shaped must be changed to desired shape"
        image, mask = self.augment_image(image, mask)
        # scale between 0 and 1
        image = image / 255.0
        # normalised images between -1 and 1
        image = (image - 0.5) / 0.5
        # convert to torch tensor
        assert (image.shape[-1] == 3)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        # coordinates
        x, y = re.search('slide\d+_core\d+_(\d+)_(\d+)', path.name).groups()
        if mask is not None:
            mask = torch.from_numpy(mask.copy()).long()
        return {
            'input': image,
            'gleason_mask': mask,
            'input_path': str(path),
            'mask_path': str(mask_path),
            'x_offset': int(x) + origin[0],
            'y_offset': int(y) + origin[1]
        } if mask is not None else {
            'input': image,
            'input_path': str(path),
            'x_offset': int(x) + origin[0],
            'y_offset': int(y) + origin[1]
        }


# script here to make joint score mask
if __name__ == '__main__':
    from pathlib import Path
    from argparse import ArgumentParser
    import multiprocessing as mp
    import re
    from tqdm import tqdm
    from imageio import imwrite

    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path, default=None)
    args = parser.parse_args()

    images_dir = args.data_dir/'Train Imgs'
    paths = sorted(images_dir.glob('*.jpg'))
    images_dir = args.data_dir/'Test'
    paths += sorted(images_dir.glob('*.jpg'))

    all_ground_truth_paths = []
    for i in range(1, 7):
        ground_truth_paths_ = sorted(Path(args.data_dir, 'physician_annotations', f'Maps{i}_T').glob('*.png'))
        ground_truth_paths = []
        for j, path in enumerate(paths):
            try:
                ground_truth_path = next(ground_truth_path for ground_truth_path in ground_truth_paths_
                                         if ground_truth_path.name.startswith(path.name[:-4]))
                ground_truth_paths.append(ground_truth_path)
            except StopIteration:
                ground_truth_paths.append(None)  # fill with none if physician hasn't annotated path image
        assert len(ground_truth_paths) == len(paths), "all paths must be matched with a path or none"
        all_ground_truth_paths.append(ground_truth_paths)

    def make_consensus_mask(path_index):
        slide_num, core_num = re.match(r'slide(\d+)_core(\d+)', paths[path_index].name).groups()
        mask_path = Path(consensus_dir / f'slide{slide_num}_core{core_num}.png')
        if mask_path.exists():
            return None
        masks = []
        for j in range(len(all_ground_truth_paths)):
            mask_path = all_ground_truth_paths[j][path_index]
            if mask_path is not None:
                masks.append(imread(mask_path))
        joint_mask = np.stack(masks, axis=0)
        mode_mask, count = mode(joint_mask, axis=0)
        imwrite(consensus_dir / f'slide{slide_num}_core{core_num}.png', mode_mask.squeeze())

    pbar = tqdm(total=len(paths), desc="Making consensus maps ...")

    def update(*a):
        pbar.update()

    consensus_dir = args.data_dir/'data'/'consensus_masks'
    consensus_dir.mkdir(exist_ok=True, parents=True)
    with mp.Pool(8) as pool:
        for path_index in range(len(paths)):
            result = pool.apply_async(make_consensus_mask, (path_index,), callback=update)
            result.get()
