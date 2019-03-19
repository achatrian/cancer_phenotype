import os
from pathlib import Path
from itertools import accumulate
import datetime
import json
from .base_dataset import BaseDataset
from utils import utils
from .wsi_reader import WSIReader


class WSIDataset(BaseDataset):

    def __init__(self, opt):
        r"""
        Simple dataset to read tiles from WSIs using WSIReader
        :param opt:
        """
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
        parser.add_argument('--slide_id', type=str, default='', help="Whether to load a single slide")
        parser.add_argument('--mpp', type=float, default=0.5, help="Target millimeter per pixel resolution to read slide")
        parser.add_argument('--qc_mpp', type=float, default=4.0, help="Target millimeter per pixel resolution for quality control on slide")
        parser.add_argument('--check_tile_blur', action='store_true', help="Reject tiles that result blurred according to my criterion")
        parser.add_argument('--check_tile_fold', action='store_true', help="Reject tiles that contain a fold according to my criterion")
        parser.add_argument('--overwrite_qc', action='store_true', help="Perform quality control on all slides and overwrite files")
        return parser

    def name(self):
        return "WSIDataset"

    def setup(self, good_files=tuple()):
        # Lazy set up as it can be slow
        # Determine tile locations in all the images
        root_path = Path(self.opt.data_dir)
        paths = list(root_path.glob('./*.svs')) + list(root_path.glob('./*/*.svs'))  # avoid expensive recursive search
        if self.opt.slide_id:
            paths = [str(path) for path in paths if self.opt.slide_id in str(path)]
        files = sorted((str(path) for path in paths), key=lambda file: os.path.basename(file))  # sorted by basename
        good_files = sorted(good_files, reverse=True)  # reversed, as last element is checked and then popped from the list
        # read quality_control results if available (last produced by date):
        qc_results = root_path.glob('data/quality_control/qc_*')
        qc_result = sorted(qc_results,
                           key=lambda qc_name: datetime.datetime.strptime(qc_name.name[3:-5], "%Y-%m-%d"))  # strip 'qc_' and '.json'
        try:
            qc_name = qc_result[-1]  # most recent quality_control
            qc_store = json.load(open(root_path/'data'/'quality_control'/qc_name, 'r'))
            print(f"Using qc results from '{str(Path(qc_name).name)}'")
        except IndexError:
            qc_store = None
        for file in files:
            if not utils.is_pathname_valid(file):
                raise FileNotFoundError("Invalid path: {}".format(file))
            name = os.path.basename(file)
            if good_files:
                if not name.startswith(good_files[-1]):  # start is used
                    continue  # keep only strings matching good_files
                else:
                    good_files.pop()
            slide = WSIReader(self.opt, file)
            slide.find_good_locations(qc_store)  # TODO this still doesn't work perfectly
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











