import os
from multiprocessing import Pool
from pathlib import Path
from itertools import accumulate
from .base_dataset import BaseDataset
from utils import utils
from .wsi_reader import WSIReader


class WSIDataset(BaseDataset):

    def __init__(self, opt):
        super(WSIDataset, self).__init__()
        self.opt = opt
        self.files = []
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

    def setup(self, good_files=tuple()):
        # Lazy set up as it can be slow
        # Determine tile locations in all the images
        root_path = Path(self.opt.data_dir)
        paths = root_path.glob('**/*.svs')
        files = sorted((str(path) for path in paths), key=lambda file: os.path.basename(file))  # sorted by basename
        good_files = sorted(good_files, reverse=True)  # reversed, as last element is checked and then popped from the list
        for file in files:
            if not utils.is_pathname_valid(file):
                raise FileNotFoundError("Invalid path: {}".format(file))
            name = os.path.basename(file)
            if good_files:
                if not name.startswith(good_files[-1]):  # start is used
                    continue  # keep only strings matching good_files
                else:
                    good_files.pop()
            self.files.append(file)
            slide = WSIReader(self.opt, file)
            slide.find_good_locations()  # TODO this still doesn't work perfectly
            self.tiles_per_slide.append(len(slide))
            self.slides.append(slide)
        self.tile_idx_per_slide = list(accumulate(self.tiles_per_slide))  # to index tiles quickly

    def __len__(self):
        return sum(self.tiles_per_slide)

    def __getitem__(self, item):
        if not self.files:
            raise ValueError("WSI Dataset is not initialized - call setup(file)")
        slide_idx = next(i for i, bound in enumerate(self.tile_idx_per_slide) if item < bound)
        slide = self.slides[slide_idx]
        tile_idx = item - self.tile_idx_per_slide[slide_idx - 1] if item > 0 else item  # index of tile in slide
        tile = slide[tile_idx]
        tile_loc = slide.good_locations[tile_idx]
        output = dict(tile=tile, location=tile_loc, file_name=slide.file_name)  # return image and relative location in slide
        return output











