import os
from pathlib import Path
from itertools import accumulate
import datetime
import json
import warnings
import random
import numpy as np
from torchvision.transforms import ToTensor
from .base_dataset import BaseDataset
from utils import utils
from .wsi_reader import WSIReader
from base.utils.annotation_builder import AnnotationBuilder


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
        self.good_files = None  # used by set up to know which image files to index

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--mpp', type=float, default=0.5, help="Target millimeter per pixel resolution to read slide")
        parser.add_argument('--qc_mpp', type=float, default=4.0, help="Target millimeter per pixel resolution for quality control on slide")
        parser.add_argument('--check_tile_blur', action='store_true', help="Reject tiles that result blurred according to my criterion")
        parser.add_argument('--check_tile_fold', action='store_true', help="Reject tiles that contain a fold according to my criterion")
        parser.add_argument('--overwrite_qc', action='store_true', help="Perform quality control on all slides and overwrite files")
        parser.add_argument('--tissue_threshold', type=float, default=0.4, help="Threshold of tissue filling in tile for it to be considered a tissue tile")
        parser.add_argument('--saturation_threshold', type=int, default=20, help="Saturation difference threshold of tilefor it to be considered a tissue tile")
        parser.add_argument('--area_annotation_dir', type=str, default='tumour_area_annotation', help="Name of subdir where delimiting area annotations can be found")
        parser.add_argument('--area_annotation_scale', type=float, default=1.0, help="If annotations are used to select subset of tiles in WSI, their contours are divided by this factor")
        parser.add_argument('--max_total_tiled_area', type=int, default=1e7, help="Upper limit to area covered by dataset tiles, in um^2")
        parser.add_argument('--max_num_subboxes', type=int, default=30, help="If max total tiled area > 0, this parameter chooses the number of boxes used to sample the tiles. More boxes means smaller boxes")
        return parser

    def name(self):
        return "WSIDataset"

    def setup(self):
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
        # good files are in reversed order, as last element is checked and then popped from the list
        good_files = sorted(self.good_files, reverse=True) if self.good_files else ()
        # read quality_control results if available (last produced by date):
        qc_results = root_path.glob('data/quality_control/qc_*')
        qc_result = sorted(qc_results,
                           key=lambda qc_name: datetime.datetime.strptime(qc_name.name[3:-5], "%Y-%m-%d"))  # strip 'qc_' and '.json'
        try:
            # TODO not using this - remove (from WSIReader as well) -
            qc_name = qc_result[-1]  # most recent quality_control
            qc_store = json.load(open(root_path/'data'/'quality_control'/qc_name, 'r'))
            print(f"Using qc results from '{str(Path(qc_name).name)}'")
        except IndexError:
            qc_store = None
        to_filter = bool(good_files)  # whether to use tuple files to filter slide files in data dir
        # perform quality control on each file and store WSIReader instances
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
            slide.find_tissue_locations(self.opt.tissue_threshold, self.opt.saturation_threshold, qc_store)
            # restrict location by tumour annotation area only
            try:  # try to read annotation i
                with open(Path(self.opt.data_dir) / 'data' / self.opt.area_annotation_dir /
                          (self.opt.slide_id + '.json'), 'r') as annotation_file:
                    annotation_obj = json.load(annotation_file)
                    annotation_obj['slide_id'] = self.opt.slide_id
                    annotation_obj['project_name'] = 'tumour_area'
                    annotation_obj['layer_names'] = ['Tumour area']
                    contours, layer_name = AnnotationBuilder.from_object(annotation_obj).\
                        get_layer_points('Tumour area', contour_format=True)
                    slide.filter_locations(contours, delimiters_scaling=self.opt.area_annotation_scale)
            except FileNotFoundError:
                warnings.warn(f"Could not load area annotation for {Path(slide.file_name).name}, returning tiles from whole image")
            if self.opt.max_total_tiled_area:
                # restrict locations by max area, by taking subboxes of slide
                box_pixel_area = self.opt.max_total_tiled_area / slide.mpp_x / slide.mpp_y / self.opt.max_num_subboxes  # convert to pixels^2 first
                boxes, iter_num, total_covered_area = [], 0, 0.0
                while len(boxes) < self.opt.max_num_subboxes and total_covered_area < self.opt.max_total_tiled_area:
                    covered_pixel_area = 0.0
                    w = np.sqrt(box_pixel_area) * (1 + (-1) ** int(random.random() > 0.5) * random.random() / 10)  # not perfectly square, add small randomness (max/min = +- 10%)
                    h = box_pixel_area / w
                    scale_up = random.random()  # works quite well to get final desired area with few boxes
                    h, w = h * (1 + scale_up), w * (1 + scale_up)
                    x, y = random.choice(slide.tissue_locations)  # choose random location as box upper left corner
                    # test if box contains tissue locations
                    for i, (xt, yt) in enumerate(slide.tissue_locations):
                        if x <= xt <= x + w and y <= yt <= y + h:
                            covered_pixel_area += self.opt.patch_size ** 2
                    if covered_pixel_area >= box_pixel_area * (0.99 ** iter_num):
                        boxes.append((x, y, w, h))
                        total_covered_area += covered_pixel_area * slide.mpp_x * slide.mpp_y
                    iter_num += 1
                slide.filter_locations(delimiters=boxes)
                print(f"{self.name()}: total area covered for {name} is {int(total_covered_area)} um^2")
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
            self.good_files = (self.opt.slide_id,)  # when .setup() is called in main script, only slide is considered
        else:
            super().make_subset(selector, selector_type, store_name)
