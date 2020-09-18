from pathlib import Path
import json
from ihc.datasets.ihcpatch_dataset import IHCPatchDataset


r"""
Additionally to training tiles, load all tiles classified as being ambiguous (i.e. part of a focus of interest) when they aren't.
"""


class IHCActive(IHCPatchDataset):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt.no_ground_truth = True  # misclassified tiles don't have ground truth masks
        if self.opt.phase == 'train':
            misclassified_paths = list(self.opt.misclassified_dir.glob('*/*.png'))
            self.paths += misclassified_paths
            with open(self.opt.misclassified_dir/'misclassified_info.json', 'r') as misclassified_file:
                self.misclassified_info_file = json.load(misclassified_file)
            self.labels += [1]*len(misclassified_paths)
            print(f"Adding {len(misclassified_paths)} misclassified tiles by experiment {self.opt.misclassified_dir.name} to dataset")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = super().modify_commandline_options(parser, is_train)
        parser.add_argument('--misclassified_dir', type=Path)
        return parser

    def name(self):
        return "IHCActiveDataset"
