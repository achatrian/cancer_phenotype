import time
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
        return parser

    def name(self):
        return "WSIDataset"

    def setup(self, files):
        # Lazy set up as it can be slow
        # Determine tile locations in all the images
        time_meter = utils.AverageMeter()
        for file in files:
            if not utils.is_pathname_valid(file):
                raise FileNotFoundError("Invalid path: {}".format(file))
            scan_start_time = time.time()
            slide = WSIReader(self.opt, file).find_good_locations()
            time_meter.update(time.time() - scan_start_time, 1)
            self.tiles_per_slide.append(len(slide))
            self.slides.append(slide)
        self.tile_idx_per_slide = list(accumulate(self.tiles_per_slide))  # to index tiles quickly
        return time_meter.avg

    def __len__(self):
        return sum(self.tiles_per_slide)

    def __getitem__(self, item):
        slide_idx = next(i for i, bound in enumerate(self.tile_idx_per_slide) if item < bound)
        slide = self.slides[slide_idx]
        tile_idx = item - self.tile_idx_per_slide[slide_idx - 1] if item > 0 else item  # index of tile in slide
        tile = slide[tile_idx]
        output = dict(tile=tile, location=slide.high_res_locations[item])  # return image and relative location in slide
        return output













