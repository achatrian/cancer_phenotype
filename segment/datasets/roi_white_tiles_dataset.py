from pathlib import Path
import numpy as np
import imageio
from skimage.color import rgba2rgb
import torch
from segment.datasets.roitiles_dataset import ROITilesDataset


class ROIWhiteTilesDataset(ROITilesDataset):

    def __init__(self, opt):
        super().__init__(opt)
        len0 = len(self)
        max_white_tile_num = round(self.opt.max_white_tiles_fraction*len0)
        max_white_tile_num = max_white_tile_num - max_white_tile_num % 2  # makes sure final corresponding mask path is loaded
        if self.opt.phase == 'train':
            self.paths.extend(path for i, path in enumerate(self.opt.white_tiles_dir.iterdir())
                              if path.name.endswith('_image.png') and i <= max_white_tile_num*2)
            self.gt_paths.extend(path.parent/path.name.replace('image', 'mask')
                                 for i, path in enumerate(self.opt.white_tiles_dir.iterdir())
                                 if path.name.endswith('_image.png') and i <= max_white_tile_num*2)
        len_with_white_tiles = len(self)
        print(f"{len_with_white_tiles - len0} background tiles added to training split")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = ROITilesDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--white_tiles_dir', type=Path, required=True)
        parser.add_argument('--max_white_tiles_fraction', type=float, default=0.2)
        return parser

    def name(self):
        return "ROIWhiteTilesDataset"

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
            if gt_path.parent == self.opt.white_tiles_dir:
                gt = gt*0  # ground truth for white tiles must be filled with 0's
            assert gt.ndim == 2, "Check gt format"
            image, gt = self.rescale(image, gt=gt)
            if not (isinstance(gt, np.ndarray) and gt.ndim > 0):
                raise ValueError("{} is not valid".format(gt_path))
            # im aug
            image, gt = self.augment_image(image, gt)
            # TODO test below !!!
            # paint labels in
            for i, (label, interval) in enumerate(self.label_interval_map.items()):
                gt[np.logical_and(gt >= interval[0], gt <= interval[1])] = i
            gt[np.logical_and(gt < 0, gt > i)] = 0
            assert (len(gt.shape) == 2)
            gt = torch.from_numpy(gt.copy()).long()  # change to FloatTensor for BCE
        else:
            image, _ = self.rescale(image)  # gt is none here
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
        if len(coords) > 4:
            coords = coords[1:3]
        data = dict(
            input=image,
            input_path=str(image_path),
            x_offset=int(coords[0]),
            y_offset=int(coords[1])
        )
        if not self.opt.no_ground_truth:
            data.update(
                target=gt,
                target_path=str(gt_path)
            )
        return data