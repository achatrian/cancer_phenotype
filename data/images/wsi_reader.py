import os
import argparse
from pathlib import Path
import re
import warnings
from typing import Tuple
import csv
import json
from numbers import Real
import datetime
import time
from openslide import OpenSlideError
import tqdm
import numpy as np
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
from .base_wsi_reader import TiffReader as TReader, IsyntaxReader as IReader, OpenSlideReader as OReader
from base.utils import debug


class AdvancedReader:

    def find_best_level(self, target_mpp):
        best_level_x = np.argmin(np.absolute(np.array(self.level_downsamples) * self.mpp_x - target_mpp))
        best_level_y = np.argmin(np.absolute(np.array(self.level_downsamples) * self.mpp_y - target_mpp))
        assert best_level_x == best_level_y, "This should be the same, unless pixel has different side lengths"
        read_level = int(best_level_x)  # from np.int64
        read_mpp = float(self.mpp_x) * self.level_downsamples[read_level]
        return read_level, read_mpp

    def __init__(self, file_name='', opt=None, set_mpp=None):
        r"""
        :param file_name: path to image file
        :param opt: options, as returned by get_reader_options (or dict)
        :param set_mpp: if no mpp metadata is available, set_mpp is assumed to be the highest resolution for the slide
        """
        file_name = str(file_name)
        super().__init__(file_name)  # initializes tiff or isyntax base readers
        if isinstance(opt, dict):
            self.opt = get_reader_options(include_path=False, args=opt)
        else:
            self.opt = opt or get_reader_options(False, True)
        self.file_name = file_name
        self.tissue_locations = []
        self.locations = []
        self.tile_info = dict()
        self.qc = None
        self.stride = None  # stride s
        self.no_resolution = False
        self.mpp_x, self.mpp_y = self.mpp[0] or set_mpp, self.mpp[1] or set_mpp
        if self.mpp_x is None:
            warnings.warn("No resolution information available")
            self.no_resolution = True  # TODO disable all resolution dependent commands
        # compute read level based on mpp
        if not self.no_resolution and hasattr(self.opt, 'qc_mpp') and hasattr(self.opt, 'mpp'):
            if self.opt.qc_mpp < self.opt.mpp:
                raise ValueError(f"Quality control must be done at an equal or greater MPP resolution ({self.opt.qc_mpp} < {self.opt.mpp})")
            self.read_level, self.read_mpp = self.find_best_level(self.opt.mpp)
            self.qc_read_level, self.qc_mpp = self.find_best_level(self.opt.qc_mpp)
        else:
            warnings.warn("No mpp or qc_mpp options - cannot perform quality control")
            self.find_tissue_locations = lambda *args: print("No mpp or qc_mpp options - cannot perform quality control")
        self.tissue_threshold = None
        self.tissue_percentage = None
        self.saturation_threshold = None

    def find_tissue_locations(self, tissue_threshold=0.4, saturation_threshold=20, qc_store=None):
        """
        Perform quality control on regions of slide by examining each tile
        :return:
        """
        tile_read_errors = []
        try:
            if self.opt.overwrite_qc:
                raise FileNotFoundError('Option: overwrite quality_control')
            self.read_locations(qc_store)
            if self.opt.verbose:
                print("Read quality_control info for {}".format(os.path.basename(self.file_name)))
        except FileNotFoundError:
            print("Finding tissue tiles ...")
            tissue_locations = []
            self.stride = round(self.opt.patch_size * self.opt.mpp / (self.mpp_x * self.level_downsamples[self.read_level]))
            assert self.stride == 1024  # TODO delete
            qc_sizes = (round(self.opt.patch_size * self.opt.mpp / (self.mpp_x * self.level_downsamples[self.qc_read_level])),) * 2  # size of read tiles at level qc_read_level
            xs = list(range(0, self.level_dimensions[0][0], self.stride))[:-1]  # dimensions = (width, height)
            ys = list(range(0, self.level_dimensions[0][1], self.stride))[:-1]
            self.tissue_threshold = tissue_threshold
            self.saturation_threshold = saturation_threshold
            coverage = 0
            total_tile_num = len(xs)*len(ys)
            with tqdm.tqdm(total=total_tile_num) as progress_bar:
                for x in xs:
                    for y in ys:
                        try:
                            tile = self.read_region((x, y), self.qc_read_level, qc_sizes)
                        except OpenSlideError as err:  # 'Cannot read raw tile' error on red regions of some images
                            tile_read_errors.append(str(err))
                            progress_bar.update()
                            continue
                        tile = np.array(tile)  # convert from PIL.Image
                        self.locations.append((x, y))
                        if not self.is_HnE(tile, threshold=tissue_threshold, sat_thresh=saturation_threshold):
                            self.tile_info[f'{x},{y}'] = 'empty'
                        elif self.opt.check_tile_blur and self.is_blurred(tile):
                            self.tile_info[f'{x},{y}'] = 'blur'
                        elif self.opt.check_tile_fold and self.is_folded(tile):
                            self.tile_info[f'{x},{y}'] = 'folded'
                        else:
                            self.tile_info[f'{x},{y}'] = 'tissue'
                            tissue_locations.append((x, y))
                            coverage += 1
                        progress_bar.update()
                self.tissue_locations = tissue_locations
                self.save_locations()  # overwrite if existing
            self.tissue_percentage = coverage / total_tile_num
            print(f"Performed quality control on {os.path.basename(self.file_name)}, {self.tissue_percentage * 100}% coverage")
        return tile_read_errors

    def save_locations(self):
        """
        Save info into tsv file
        :return:
        """
        name_ext = re.sub('\.(ndpi|svs)', '.json', os.path.basename(self.file_name))
        slide_loc_path = Path(self.opt.data_dir)/'data'/'quality_control'/name_ext
        Path(self.opt.data_dir, 'data', 'quality_control').mkdir(parents=True, exist_ok=True)
        quality_control = {
            'date': str(datetime.datetime.now()),
            'slide_id': Path(self.file_name).name,
            'mpp_x': self.mpp_x,
            'mpp_y': self.mpp_y,
            'qc_read_level': self.qc_read_level,
            'qc_mpp': self.qc_mpp,
            'patch_size': self.opt.patch_size,
            'tissue_threshold': self.tissue_threshold,
            'saturation_threshold': self.saturation_threshold,
            'tissue_percentage': self.tissue_percentage,
            'tissue_locations': self.tissue_locations,
            'tile_info': {f'{x},{y}': self.tile_info[f'{x},{y}'] for x, y in self.locations}
        }
        with open(slide_loc_path, 'w') as slide_loc_file:
            json.dump(quality_control, slide_loc_file)

    def read_locations(self, qc_store=None):
        """
        Attempt to load quality_control info from argument or from .tsv file
        :param qc_store:
        :return:
        """
        name_ext = re.sub('\.(ndpi|svs)', '.json', os.path.basename(self.file_name))
        if qc_store:
            qc_opt = qc_store[0]
            for option, value in qc_opt.items():
                qc_value = getattr(self.opt, option)
                if qc_value != value:  # ensure that quality_control data is same as session options
                    raise ValueError("Option mismatch between quality control and session - option: {} ({} != {})".format(
                        option, qc_value, value
                    ))
            qc_data = qc_store[1]
            for loc, info in qc_data[name_ext.split('.')[0]].items():
                self.tile_info[loc] = info
                loc = tuple(int(d) for d in loc.split(','))
                self.locations.append(loc)
                if info == 'tissue':
                    self.tissue_locations.append(loc)
        else:  # create slide file if it doesn't exist
            slide_loc_path = os.path.join(self.opt.data_dir, 'data', 'quality_control', name_ext)
            with open(slide_loc_path, 'r') as slide_loc_file:
                if os.stat(slide_loc_path).st_size == 0:  # rewrite if file is empty
                    raise FileNotFoundError(f"Empty file: {slide_loc_path}")
                quality_control = json.load(slide_loc_file)
                if quality_control['mpp_x'] != self.mpp_x or quality_control['mpp_y'] != self.mpp_y or \
                        quality_control['qc_read_level'] != self.qc_read_level or quality_control['qc_mpp'] != self.qc_mpp:
                    raise ValueError(f"Mismatching slide and quality control information for {Path(self.file_name).name} (slide/file) "
                                     f"mpp_x: {self.mpp_x:.2f}/{quality_control['mpp_x']:.2f}; mpp_y: {self.mpp_y:.2f}/{quality_control['mpp_y']:.2f}; "
                                     f"qc_mpp: {self.qc_mpp:.2f}/{quality_control['qc_mpp']:.2f}; qc_read_level: {self.qc_read_level:.2f}/{quality_control['qc_read_level']:.2f}")
                if quality_control['patch_size'] != self.opt.patch_size:
                    warnings.warn(f"Using different patch size ({self.opt.patch_size}) from quality control patch size ({quality_control['patch_size']})")
                self.tissue_locations = quality_control['tissue_locations']
                self.tile_info = quality_control['tile_info']

    def __len__(self):
        return len(self.tissue_locations)

    def __getitem__(self, item):
        r"""Returns tissue tiles"""
        return self.read_region(self.tissue_locations[item], self.read_level, (self.opt.patch_size,) * 2)

    def filter_locations(self, delimiters, delimiters_scaling=None, loc_tol=0.2):
        r"""
        Select subset of locations based on a spatial delimiter
        :param delimiters: seq of bounding boxes = (x, y, w, h), ... or seq of contours [[[0, 0], [0, 1]]], ...
        :param delimiters_scaling: divide delimiters by this scaling
        :param loc_tol: percentage tolerance of origin position, to account for large tile quality control
        """
        if delimiters_scaling:
            delimiters = [(np.array(delimiter) / delimiters_scaling).astype(np.int32)
                          for delimiter in delimiters if len(delimiter) > 0]
        # check is done on first element of delimiters seq only
        if len(delimiters[0]) == 4 and all(isinstance(dim, Real) for dim in delimiters[0]):
            # Bounding box delimiters
            internal_locations = set()
            for delimiter in delimiters:
                x, y, w, h = delimiter
                for i, (xt, yt) in enumerate(self.tissue_locations):
                    if x <= xt <= x + w and y <= yt <= y + h:
                        internal_locations.add((xt, yt))
        elif isinstance(delimiters[0], np.ndarray) and delimiters[0].ndim == 3:
            # Contour delimiters
            internal_locations = []
            for i, (x, y) in enumerate(self.tissue_locations):
                tentative_x, tentative_y = np.meshgrid(
                    np.linspace(x - loc_tol*self.opt.patch_size, x + loc_tol*self.opt.patch_size, 5, dtype=int),
                    np.linspace(y - loc_tol*self.opt.patch_size, y + loc_tol*self.opt.patch_size, 5, dtype=int),
                )
                tentative_origins = [(int(x), int(y)) for x, y in zip(tentative_x.ravel(), tentative_y.ravel())]
                if any(cv2.pointPolygonTest(tumour_area, origin_corner, measureDist=False) >= 0
                       for origin_corner in tentative_origins for tumour_area in delimiters):
                    internal_locations.append((x, y))
        else:
            raise ValueError(f"Delimiters needs to be sequence of bounding boxes or contours, not '{type(delimiters[0])}'")
        if not internal_locations:
            warnings.warn(f"Filtering resulted in empty reader for {Path(self.file_name).name}")
        self.tissue_locations = tuple(internal_locations)

    def __str__(self):
        return os.path.basename(self.file_name) + '\t' + '\t'.join(['{},{}'.format(x, y) for x, y in self.tissue_locations])

    def __repr__(self):
        return str(self)

    @staticmethod
    def is_HnE(image, threshold=0.4, sat_thresh=20, min_size=50, area_threshold=200):
        """Returns true if slide contains tissue or just background
        Problem: doesn't work if tile is completely white because of normalization step"""
        assert np.issubdtype(image.dtype, np.integer), "RGB representation must be integers from 0 to 255 "
        if image.shape[-1] == 4:
            #image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            image = image[..., :3]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
        # saturation check
        sat_mean = hsv_image[..., 1].mean()
        empirical = hsv_image.prod(axis=2)  # found by Ka Ho to work
        empirical = empirical/np.max(empirical)*255  # found by Ka Ho to work
        kernel = np.ones((20, 20), np.uint8)
        ret, _ = cv2.threshold(empirical.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx((empirical > ret).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = remove_small_holes(mask.astype(bool), area_threshold=area_threshold, connectivity=3)
        mask = remove_small_objects(mask.astype(bool), min_size=min_size)
        return mask.mean() > threshold and sat_mean >= sat_thresh

    @staticmethod
    def is_blurred(image):
        # TODO implement
        return False

    @staticmethod
    def is_folded(image):
        # TODO implement
        return False


class WSIReader(AdvancedReader, TReader):

    def __init__(self, file_name='', opt=None, set_mpp=None):
        super().__init__(file_name, opt, set_mpp)


class OpenSlideReader(AdvancedReader, OReader):

    def __init__(self, file_name='', opt=None, set_mpp=None):
        super().__init__(file_name, opt, set_mpp)


class IsyntaxReader(AdvancedReader, IReader):

    def __init__(self, file_name='', opt=None, set_mpp=None):
        super().__init__(file_name, opt, set_mpp)


def make_wsi_reader(file_name='', opt=None, set_mpp=None, openslide=False):
    if Path(file_name).suffix == '.isyntax':
        return IsyntaxReader(file_name, opt, set_mpp)
    elif Path(file_name).suffix == '.dzi':
        from dzi_io import DZIIO
        return DZIIO(file_name)  # opt does nothing here
    elif openslide:
        return OpenSlideReader(file_name, opt, set_mpp)
    else:
        return WSIReader(file_name, opt, set_mpp)


def add_reader_args(parser, include_path=False, include_thresholds=True):
    def get_option_strings(parser: argparse.ArgumentParser):
        option_strings = set()
        for action in parser._actions:
            for option_string in action.option_strings:
                option_strings.add(option_string)
        return option_strings

    preexisting_options = get_option_strings(parser)
    if include_path:
        parser.add_argument('slide_path', type=str, default='')
    parser.add_argument('--qc_mpp', default=2.0, type=float, help="MPP value to perform quality control on slide")
    if '--mpp' not in preexisting_options:
        parser.add_argument('--mpp', default=0.40, type=float, help="MPP value to read images from slide")   # CHANGED DEFAULT FROM 0.5 to 0.4
    if '--data_dir' not in preexisting_options:
        parser.add_argument('--data_dir', type=str, default='', help="Dir where to save qc result")
    parser.add_argument('--check_tile_blur', action='store_true', help="Check for blur")
    parser.add_argument('--check_tile_fold', action='store_true', help="Check tile fold")
    parser.add_argument('--overwrite_qc', action='store_true', help="Overwrite saved quality control data")
    if '--patch_size' not in preexisting_options:
        parser.add_argument('--patch_size', type=int, default=1024, help="Pixel size of patches (at desired resolution)")
    if '--verbose' not in preexisting_options:
        parser.add_argument('--verbose', action='store_true', help="Print more information")
    if include_thresholds:
        parser.add_argument('--tissue_threshold', type=float, default=0.4,
                            help="Threshold of tissue filling in tile for it to be considered a tissue tile")
        parser.add_argument('--saturation_threshold', type=int, default=25,
                            help="Saturation difference threshold of tile for it to be considered a tissue tile")
    return parser


def get_reader_options(include_path=True, include_thresholds=True, args=()):
    parser = argparse.ArgumentParser(usage="base_wsi_reader.py path_to_image [options]" if include_path else None)
    parser = add_reader_args(parser, include_path, include_thresholds)
    if args:
        if isinstance(args, dict):
            args = tuple(f'--{key}={value}' for key, value in args.items())
            args = ('--overwrite_qc',) + args
        opt, unknown = parser.parse_known_args(args)
    else:
        opt, unknown = parser.parse_known_args()
    return opt


    # def export_tissue_tiles(self, tile_dir='tiles', export_delimiters=(), delimiters_scaling=None):
    #     r"""
    #     Save quality-controlled tiles
    #     :param tile_dir:
    #     :param export_delimiters:
    #     :param delimiters_scaling:
    #     :return:
    #     """
    #     sizes = (self.opt.patch_size, )*2
    #     save_tiles_dir = Path(self.opt.data_dir)/'data'/tile_dir/Path(self.file_name).name[:-4]  # remove extension
    #     utils.mkdirs(save_tiles_dir)
    #     with open(save_tiles_dir/'resolution.json', 'w') as res_file:
    #         json.dump({
    #             'target_mpp': self.opt.mpp,
    #             'target_qc_mpp': self.opt.qc_mpp,
    #             'read_level': self.read_level,
    #             'qc_read_level': self.qc_read_level,
    #             'read_mpp': self.read_mpp,
    #             'qc_mpp': self.qc_mpp,
    #             'tissue_locations': self.tissue_locations
    #         },
    #                   res_file)
    #     if self.tissue_locations:
    #         print("Begin exporting tiles ...")
    #         log_freq, start_time, last_print = 5, time.time(), 0
    #         exported_locations = []
    #         with open(save_tiles_dir / 'log.txt', 'w') as log_file:  # 'w' flags overwrites file completely
    #             log_file.write(f"{''.join(datetime.datetime.now().__str__().split(' ')[0:2])} - begin export ...\n")
    #             if export_delimiters:
    #                 self.filter_locations(export_delimiters, delimiters_scaling)  # TODO test
    #             for x, y in tqdm.tqdm(self.tissue_locations):
    #                 tile = self.read_region((x, y), self.read_level, sizes)
    #                 tile = np.array(tile)
    #                 imageio.imwrite(save_tiles_dir/f'{x}_{y}.png', tile)
    #                 exported_locations.append((x, y))
    #                 if len(exported_locations) % log_freq == 0 and last_print != len(exported_locations):
    #                     log_file.write(f"({len(exported_locations)}: {time.time() - start_time:.3f}s) Exported {len(exported_locations)} tiles ...\n")
    #                     last_print = len(exported_locations)
    #             if not exported_locations:
    #                 warnings.warn(f"No tissue locations overlapped with annotation - {Path(self.file_name).name}")
    #             log_file.write(f"{time.time() - start_time:.3f}s Done !")
    #             print(f"Exported {len(exported_locations)} tiles.")
    #     else:
    #         warnings.warn(f"No locations to export - {Path(self.file_name).name}")


    # def export_tissue_tiles(self, tile_dir='tiles', export_delimiters=(), delimiters_scaling=None):
    #     r"""
    #     Save quality-controlled tiles
    #     :param tile_dir:
    #     :param export_delimiters:
    #     :param delimiters_scaling:
    #     :return:
    #     """
    #     sizes = (self.opt.patch_size, )*2
    #     save_tiles_dir = Path(self.opt.data_dir)/'data'/tile_dir/Path(self.file_name).name[:-4]  # remove extension
    #     utils.mkdirs(save_tiles_dir)
    #     with open(save_tiles_dir/'resolution.json', 'w') as res_file:
    #         json.dump({
    #             'target_mpp': self.opt.mpp,
    #             'target_qc_mpp': self.opt.qc_mpp,
    #             'read_level': self.read_level,
    #             'qc_read_level': self.qc_read_level,
    #             'read_mpp': self.read_mpp,
    #             'qc_mpp': self.qc_mpp,
    #             'tissue_locations': self.tissue_locations
    #         },
    #                   res_file)
    #     if self.tissue_locations:
    #         print("Begin exporting tiles ...")
    #         log_freq, start_time, last_print = 5, time.time(), 0
    #         exported_locations = []
    #         with open(save_tiles_dir / 'log.txt', 'w') as log_file:  # 'w' flags overwrites file completely
    #             log_file.write(f"{''.join(datetime.datetime.now().__str__().split(' ')[0:2])} - begin export ...\n")
    #             if export_delimiters:
    #                 self.filter_locations(export_delimiters, delimiters_scaling)  # TODO test
    #             for x, y in tqdm.tqdm(self.tissue_locations):
    #                 tile = self.read_region((x, y), self.read_level, sizes)
    #                 tile = np.array(tile)
    #                 imageio.imwrite(save_tiles_dir/f'{x}_{y}.png', tile)
    #                 exported_locations.append((x, y))
    #                 if len(exported_locations) % log_freq == 0 and last_print != len(exported_locations):
    #                     log_file.write(f"({len(exported_locations)}: {time.time() - start_time:.3f}s) Exported {len(exported_locations)} tiles ...\n")
    #                     last_print = len(exported_locations)
    #             if not exported_locations:
    #                 warnings.warn(f"No tissue locations overlapped with annotation - {Path(self.file_name).name}")
    #             log_file.write(f"{time.time() - start_time:.3f}s Done !")
    #             print(f"Exported {len(exported_locations)} tiles.")
    #     else:
    #         warnings.warn(f"No locations to export - {Path(self.file_name).name}")






