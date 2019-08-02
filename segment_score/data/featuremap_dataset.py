from pathlib import Path
import re
import json
import torch
from base.data.base_dataset import BaseDataset


class FeatureMapDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        pattern = re.compile('(\d{0,8})_(\d{0,8})\.pt')  # .pt is the extension for pytorch tensors
        feature_maps_dir = Path(opt.data_dir)/'data'/'feature_maps'
        self.paths = tuple(path for path in feature_maps_dir.iterdir() if bool(pattern.match(path.name)))
        assert self.paths, r"Paths tuple cannot be empty"
        with open(feature_maps_dir/'dice_values.json', 'r') as dice_values_file:
            self.labels = json.load(dice_values_file)
        assert self.labels, r"Labels tuple cannot be empty"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        name = path.name[:-3]  # remove pickle extension .pt
        feature_map = torch.load(path).float()
        label = self.labels[name]
        return feature_map, label






