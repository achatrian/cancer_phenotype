import os
from pathlib import Path
from itertools import accumulate
import datetime
import json
import warnings
import numpy as np
from torchvision.transforms import ToTensor
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
        self.to_tensor = ToTensor()
        if self.opt.is_apply and self.opt.workers > 0:
            warnings.warn("WSIReader is subclassed from OpenSlide, which has ctypes objects containing pointers. Since these cannot be pickled, dataloader breaks.")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--mpp', type=float, default=0.5, help="Target millimeter per pixel resolution to read slide")
        parser.add_argument('--qc_mpp', type=float, default=4.0, help="Target millimeter per pixel resolution for quality control on slide")
        parser.add_argument('--check_tile_blur', action='store_true', help="Reject tiles that result blurred according to my criterion")
        parser.add_argument('--check_tile_fold', action='store_true', help="Reject tiles that contain a fold according to my criterion")
        parser.add_argument('--overwrite_qc', action='store_true', help="Perform quality control on all slides and overwrite files")
        return parser

    def name(self):
        return "WSIDataset"

    def setup(self, good_files=tuple()):
        # Reset if setup is done again
        self.slides = []
        self.tiles_per_slide = []
        self.tile_idx_per_slide = []
        self.metadata_fields = []
        self.metadata = None
        # Lazy set up as it can be slow
        # Determine tile locations in all the images
        root_path = Path(self.opt.data_dir)
        paths = list(root_path.glob('./*.svs')) + list(root_path.glob('./*/*.svs')) + \
                list(root_path.glob('./*.ndpi')) + list(root_path.glob('./*/*.ndpi'))  # avoid expensive recursive search
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
        to_filter = bool(good_files)  # whether to use tuple files to filter slide files in data dir
        for file in files:
            if not utils.is_pathname_valid(file):
                raise FileNotFoundError("Invalid path: {}".format(file))
            name = os.path.basename(file)
            if to_filter:
                if len(good_files) == 0:
                    break
                if not name.startswith(good_files[-1]):  # start is used
                    continue  # keep only strings matching good_files
                good_files.pop()
            slide = WSIReader(self.opt, file)
            slide.find_tissue_locations(qc_store)
            self.tiles_per_slide.append(len(slide))
            self.slides.append(slide)
        self.tile_idx_per_slide = list(accumulate(self.tiles_per_slide))  # to index tiles quickly

    def __len__(self):
        return sum(self.tiles_per_slide)

    def __getitem__(self, item):
        if not self.slides:
            raise ValueError("WSI Dataset is not initialized - call setup(file)")
        slide_idx = next(i for i, bound in enumerate(self.tile_idx_per_slide) if item < bound)
        slide = self.slides[slide_idx]
        tile_idx = item - self.tile_idx_per_slide[slide_idx - 1] if item > 0 else item  # index of tile in slide
        tile = slide[tile_idx]
        tile = np.array(tile.convert('RGB'))  # if RGBA, convert to RGB
        tile = tile / 255.0  # scale between 0 and 1
        tile = (tile - 0.5) / 0.5  # normalised image between -1 and 1
        tile_loc = slide.tissue_locations[tile_idx]
        output = dict(input=self.to_tensor(tile).float(),
                      x_offset=tile_loc[0],
                      y_offset=tile_loc[1],
                      input_path=slide.file_name,
                      read_mpp=slide.read_mpp,
                      base_mpp=float(slide.properties[slide.PROPERTY_NAME_MPP_X]))  # return image and relative location in slide
        return output

    def make_subset(self, selector='', selector_type='match', store_name='paths'):
        if hasattr(self.opt, 'slide_id'):
            self.setup(good_files=(self.opt.slide_id,))
