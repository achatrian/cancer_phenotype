from pathlib import Path
from itertools import accumulate
from .base_dataset import BaseDataset
from utils import utils
from .wsi_reader import WSIReader


class WSIDataset(BaseDataset):

    def __init__(self, opt):
        super(WSIDataset, self).__init__()
        self.opt = opt
        self.slides = []
        self.tiles_per_slide = []
        self.tile_idx_per_slide = []
        self.metadata_fields = []
        self.metadata = None

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--wsi_read_level', type=int, default=2, help="Resolution at which slides are read")
        parser.add_argument('--check_tile_blur', action='store_true', help="Reject tiles that result blurred according to my criterion")
        parser.add_argument('--check_tile_fold', action='store_true', help="Reject tiles that contain a fold according to my criterion")
        return parser

    def name(self):
        return "WSIDataset"

    def setup(self):
        # Lazy set up as it can be slow
        # Determine tile locations in all the images
        root_path = Path(self.opt.dataroot)
        paths = root_path.glob('**/*.svs')
        self.files = sorted(str(path) for path in paths)
        for file in self.files:
            if not utils.is_pathname_valid(file):
                raise FileNotFoundError("Invalid path: {}".format(file))
            slide = WSIReader(self.opt, file).find_good_locations()
            self.tiles_per_slide.append(len(slide))
            self.slides.append(slide)
        self.tile_idx_per_slide = list(accumulate(self.tiles_per_slide))  # to index tiles quickly

    def __len__(self):
        return sum(self.tiles_per_slide)

    def __getitem__(self, item):
        slide_idx = next(i for i, bound in enumerate(self.tile_idx_per_slide) if item < bound)
        slide = self.slides[slide_idx]
        tile_idx = item - self.tile_idx_per_slide[slide_idx - 1] if item > 0 else item  # index of tile in slide
        tile = slide[tile_idx]
        output = dict(tile=tile, location=slide.high_res_locations[item])  # return image and relative location in slide
        return output













