from base.data.base_dataset import BaseDataset, get_augment_seq, RandomCrop


class DZISegDataset(BaseDataset):
    r""" Dataset to apply segmentation results to DZI image
    """

    def __init__(self, opt):
        super(DZISegDataset, self).__init__()
        self.opt = opt

    def __len__(self):
        return len(self.paths)

    def name(self):
        return "TileSegDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def setup(self):
        pass

    def __getitem__(self, item):
        pass
