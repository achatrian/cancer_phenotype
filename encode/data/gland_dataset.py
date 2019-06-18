from itertools import accumulate, chain
from pathlib import Path
from torchvision.transforms import ToTensor
from base.data.base_dataset import BaseDataset
from quant import read_annotations


class GlandDataset(BaseDataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.slides_paths = [path for path in Path(self.opt.data_dir).iterdir() if path.suffix in ('.svs', '.ndpi')]
        glands_tiles_path = Path(self.opt.data_dir)/'data'/'tiles'/'glands'
        self.paths = [path for path in glands_tiles_path.glob('*/*_gland*.png')]
        assert self.paths, "Paths must not be empty"
        contour_struct = read_annotations(Path(self.opt.data_dir))
        self.to_tensor = ToTensor()

    def name(self):
        return "GlandDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __len__(self):
        return 0

    def __getitem__(self, item):
        pass


