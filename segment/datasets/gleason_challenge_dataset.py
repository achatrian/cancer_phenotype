from imageio import imread
import numpy as np
import cv2
import torch
from scipy.stats import mode
from base.datasets.base_dataset import BaseDataset, RandomCrop, get_augment_seq
from base.utils import debug


class GleasonChallengeDataset(BaseDataset):
    r""""""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.paths = []
        self.opt.data_dir = Path(self.opt.data_dir)
        self.images_dir = self.opt.data_dir/('Train Imgs' if self.opt.phase == 'train' else 'Test')
        self.paths = sorted(self.images_dir.glob('*.jpg'))
        # read annotations from all physicians
        for i in range(1, 7):
            ground_truth_paths_ = sorted(Path(self.opt.data_dir, 'physician_annotations', f'Maps{i}_T').glob('*.png'))
            ground_truth_paths = []
            for j, path in enumerate(self.paths):
                try:
                    ground_truth_path = next(ground_truth_path for ground_truth_path in ground_truth_paths_
                                             if ground_truth_path.name.startswith(path.name[:-4]))
                    ground_truth_paths.append(ground_truth_path)
                except StopIteration:
                    ground_truth_paths.append(None)  # fill with none if physician hasn't annotated path image
            assert len(ground_truth_paths) == len(self.paths), "all paths must be matched with a path or none"
            setattr(self, f'ground_truth_paths{i}', ground_truth_paths)
        # compute consensus masks
        consensus_mask_dir = Path(self.opt.data_dir/'data'/'consensus_masks')
        try:
            self.consensus_mask_paths = list(consensus_mask_dir.iterdir())
        except FileNotFoundError:
            print("Consensus masks have not been created")
            raise

        self.random_crop = RandomCrop(self.opt.patch_size)
        if self.opt.augment_level:
            self.aug_seq = get_augment_seq(opt.augment_level)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--zoom_factor', type=float, default=1.0)
        return parser

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        mask_path = self.consensus_mask_paths[item]
        image, mask = imread(path), imread(mask_path)
        # rescale according to the zoom factor
        image = cv2.resize(image, None, fx=self.opt.zoom_factor, fy=self.opt.zoom_factor, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, None, fx=self.opt.zoom_factor, fy=self.opt.zoom_factor, interpolation=cv2.INTER_NEAREST)
        # pad/crop if needed
        if image.shape[0] < self.opt.patch_size:
            delta = (self.opt.patch_size - image.shape[0])//2
            image = cv2.copyMakeBorder(delta, delta, delta, delta, delta, cv2.BORDER_REFLECT)
            mask = cv2.copyMakeBorder(delta, delta, delta, delta, delta, cv2.BORDER_REFLECT)
        else:
            cat = np.concatenate((image, mask[..., np.newaxis]), axis=2)
            if self.opt.phase == 'test':
                cat = self.random_crop(cat)
            image, mask = cat[..., :3], cat[..., 4].squeeze()
        if self.opt.augment_level:
            seq_det = self.aug_seq.to_deterministic()
            image = seq_det.augment_image(image)
            mask = np.squeeze(seq_det.augment_image(np.tile(mask[..., np.newaxis], (1, 1, 3)), ground_truth=True))
            mask = mask[..., 0]
        # scale between 0 and 1
        image = image / 255.0
        # normalised images between -1 and 1
        image = (image - 0.5) / 0.5
        # convert to torch tensor
        assert (image.shape[-1] == 3)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).float()
        return {
            'input': image,
            'target': mask,
            'input_path': path,
            'target_path': mask_path
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
        slide_num, core_num = re.match(r'slide(\d+)_core(\d+)', paths[i].name).groups()
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

    consensus_dir = args.data_dir / 'data' / 'consensus_masks'
    consensus_dir.mkdir(exist_ok=True, parents=True)
    with mp.Pool(8) as pool:
        for path_index in range(len(paths)):
            result = pool.apply_async(make_consensus_mask, (path_index,), callback=update)
            result.get()


