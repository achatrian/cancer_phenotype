import os
from pathlib import Path
import re
import warnings
import csv
import json
import tqdm
import datetime
import time
from openslide import OpenSlide, PROPERTY_NAME_MPP_X, PROPERTY_NAME_MPP_Y, PROPERTY_NAME_BACKGROUND_COLOR
import numpy as np
import cv2
from skimage.morphology import remove_small_objects
import imageio
from base.utils import utils


class WSIReader(OpenSlide):

    def __init__(self, opt, file_name):
        file_name = str(file_name)
        super(WSIReader, self).__init__(file_name)
        slide_format = WSIReader.detect_format(file_name)
        if not slide_format:
            warnings.warn("Format vendor is not specified in metadata for {}".format(file_name), UserWarning)
        self.opt = opt
        self.file_name = file_name
        self.tissue_locations = []
        self.locations = []
        self.tile_info = dict()
        self.qc = None
        # compute read level based on mpp
        if self.opt.qc_mpp < self.opt.mpp:
            raise ValueError(f"Quality control must be done at an equal or greater MPP resolution ({self.opt.qc.mpp} < {self.opt.mpp})")
        mpp_x, mpp_y = float(self.properties[PROPERTY_NAME_MPP_X]),\
                       float(self.properties[PROPERTY_NAME_MPP_Y])
        best_level_x = np.argmin(np.absolute(np.array(self.level_downsamples) * mpp_x - self.opt.mpp))
        best_level_y = np.argmin(np.absolute(np.array(self.level_downsamples) * mpp_y - self.opt.mpp))
        assert best_level_x == best_level_y; "This should be the same, unless pixel has different side lengths"
        self.read_level = int(best_level_x)  # from np.int64
        # same for quality_control read level
        best_level_qc_x = np.argmin(
            np.absolute(np.array(self.level_downsamples) * mpp_x - self.opt.qc_mpp)
        )
        best_level_qc_y = np.argmin(
            np.absolute(np.array(self.level_downsamples) * mpp_y - self.opt.qc_mpp))
        assert best_level_qc_x == best_level_qc_y; "This should be the same, unless pixel has different side lengths"
        self.qc_read_level = int(best_level_qc_x)

    def find_tissue_locations(self, qc_store=None):
        """
        Perform quality control on regions of slide by examining each tile
        :return:
        """
        try:
            if self.opt.overwrite_qc:
                raise FileNotFoundError('Option: overwrite quality_control')
            self.read_locations(qc_store)
            if self.opt.verbose:
                print("Read quality_control info for {}".format(os.path.basename(self.file_name)))
        except FileNotFoundError as err:
            print("Finding tissue tiles ...")
            tissue_locations = []
            qc_downsample = int(self.level_downsamples[self.qc_read_level])
            read_downsample = int(self.level_downsamples[self.read_level])
            qc_stride = self.opt.patch_size * read_downsample  # size of read tiles at level 0
            qc_sizes = (qc_stride // qc_downsample,) * 2  # size of read tiles at level qc_read_level
            xs = list(range(0, self.level_dimensions[0][0], qc_stride))  # dimensions = (width, height)
            ys = list(range(0, self.level_dimensions[0][1], qc_stride))
            with tqdm.tqdm(total=len(xs)*len(ys)) as progress_bar:
                for x in xs:
                    for y in ys:
                        tile = self.read_region((x, y), self.qc_read_level, qc_sizes)
                        tile = np.array(tile)  # convert from PIL.Image
                        self.locations.append((x, y))
                        if not self.is_HnE(tile):
                            self.tile_info[f'{x},{y}'] = 'empty'
                        elif self.opt.check_tile_blur and self.is_blurred(tile):
                            self.tile_info[f'{x},{y}'] = 'blur'
                        elif self.opt.check_tile_fold and self.is_folded(tile):
                            self.tile_info[f'{x},{y}'] = 'folded'
                        else:
                            self.tile_info[f'{x},{y}'] = 'tissue'
                            tissue_locations.append((x, y))
                        progress_bar.update()
                self.tissue_locations = tissue_locations
                self.save_locations()  # overwrite if existing
                if self.opt.verbose:
                    print("Performed quality control on {}".format(os.path.basename(self.file_name)))

    def save_locations(self):
        """
        Save info into tsv file
        :return:
        """
        name_ext = re.sub('\.(ndpi|svs)', '.tsv', os.path.basename(self.file_name))
        slide_loc_path = os.path.join(self.opt.data_dir, 'data', 'quality_control', name_ext)
        utils.mkdirs(os.path.join(self.opt.data_dir, 'data', 'quality_control'))
        with open(slide_loc_path, 'w') as slide_loc_file:
            writer = csv.writer(slide_loc_file, delimiter='\t')
            for x, y in self.locations:
                loc = f'{x},{y}'
                writer.writerow((loc, self.tile_info[loc]))

    def read_locations(self, qc_store=None):
        """
        Attempt to load quality_control info from argument or from .tsv file
        :param qc_store:
        :return:
        """
        name_ext = re.sub('\.(ndpi|svs)', '.tsv', os.path.basename(self.file_name))
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
                reader = csv.reader(slide_loc_file, delimiter='\t')
                for loc, info in reader:
                    self.tile_info[loc] = info
                    loc = tuple(int(d) for d in loc.split(','))
                    self.locations.append(loc)
                    if info == 'tissue':
                        self.tissue_locations.append(loc)

    def __len__(self):
        return len(self.tissue_locations)

    def __getitem__(self, item):
        return self.read_region(self.tissue_locations[item], self.read_level, (self.opt.patch_size, ) * 2)

    def export_tissue_tiles(self, tile_dir='tiles', export_contours=(), contour_scaling=None):
        r"""
        Save quality-controlled tiles
        :param tile_dir:
        :param export_contours:
        :param contour_scaling:
        :return:
        """
        sizes = (self.opt.patch_size, )*2
        save_tiles_dir = Path(self.opt.data_dir)/'data'/tile_dir/Path(self.file_name).name[:-4]  # remove extension
        utils.mkdirs(save_tiles_dir)
        with open(save_tiles_dir/'resolution.json', 'w') as res_file:
            json.dump({
                'target_mpp': self.opt.mpp,
                'target_qc_mpp': self.opt.qc_mpp,
                'read_level': self.read_level,
                'qc_read_level': self.qc_read_level,
                'read_mpp': float(self.properties[PROPERTY_NAME_MPP_X]) * self.level_downsamples[self.read_level],
                'qc_mpp': float(self.properties[PROPERTY_NAME_MPP_X]) * self.level_downsamples[self.qc_read_level],
                'tissue_locations': self.tissue_locations
            },
                      res_file)
        if self.tissue_locations:
            print("Begin exporting tiles ...")
            log_freq, start_time, last_print = 5, time.time(), 0
            exported_locations = []
            if export_contours and contour_scaling:
                    export_contours = [(contour / contour_scaling).astype(np.int32) for contour in export_contours if contour.size]
            with open(save_tiles_dir / 'log.txt', 'w') as log_file:  # 'w' flags overwrites file completely
                log_file.write(f"{''.join(datetime.datetime.now().__str__().split(' ')[0:2])} - begin export ...\n")
                for x, y in tqdm.tqdm(self.tissue_locations):
                    if export_contours:
                        origin_corner = (int(x), int(y))
                        #opposite_corner = (int(origin_corner[0] + self.opt.patch_size), int(origin_corner[1] + self.opt.patch_size))
                        if any(cv2.pointPolygonTest(tumour_area, origin_corner, measureDist=False) >= 0 for tumour_area in export_contours):
                            tile = self.read_region((x, y), self.read_level, sizes)
                            tile = np.array(tile)
                            imageio.imwrite(save_tiles_dir / f'{x}_{y}.png', tile)
                            exported_locations.append(origin_corner)
                    else:
                        tile = self.read_region((x, y), self.read_level, sizes)
                        tile = np.array(tile)
                        imageio.imwrite(save_tiles_dir/f'{x}_{y}.png', tile)
                        exported_locations.append((x, y))
                    if len(exported_locations) % log_freq == 0 and last_print != len(exported_locations):
                        log_file.write(f"({len(exported_locations)}: {time.time() - start_time:.3f}s) Exported {len(exported_locations)} tiles ...\n")
                        last_print = len(exported_locations)
                if not exported_locations:
                    warnings.warn(f"No tissue locations overlapped with annotation - {Path(self.file_name).name}")
                log_file.write(f"{time.time() - start_time:.3f}s Done !")
                print(f"Exported {len(exported_locations)} tiles.")
        else:
            warnings.warn(f"No locations to export - {Path(self.file_name).name}")

    def __repr__(self):
        return os.path.basename(self.file_name) + '\t' + '\t'.join(['{},{}'.format(x, y) for x, y in self.tissue_locations])

    @staticmethod
    def save_quality_control_to_json(opt, files):
        r"""Translate all quality control results into a single json file and save it with date"""
        to_json = {}
        for file in files:
            name_ext = re.sub('\.(ndpi|svs)', '.tsv', os.path.basename(file))  # change extension
            slide_loc_path = str(Path(opt.data_dir) / 'data' / 'quality_control' / name_ext)
            with open(slide_loc_path, 'r') as slide_loc_file:
                reader = csv.reader(slide_loc_file, delimiter='\t')
                slide_qc = {loc: info for loc, info in reader}
            to_json[name_ext.split('.')[0]] = slide_qc  # can't have periods in JSON keys
        to_json = [{'wsi_read_level': opt.wsi_read_level,
                    'check_tile_blur': opt.check_tile_blur,
                    'check_tile_fold': opt.check_tile_fold,
                    'patch_size': opt.patch_size},
                   to_json]
        json.dump(to_json, open(Path(opt.data_dir) / 'data' / 'quality_control' / 'qc_{}.json'.format(
            datetime.datetime.now().__str__().split(' ')[0]
        ), 'w'))

    @staticmethod
    def is_HnE(image, threshold=0.5, sat_thresh=30, small_obj_size_factor=1/5):
        """Returns true if slide contains tissue or just background
        Problem: doesn't work if tile is completely white because of normalization step"""
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
        # saturation check
        sat_mean = hsv_image[..., 1].mean()
        empirical = hsv_image.prod(axis=2)  # found by Ka Ho to work
        empirical = empirical/np.max(empirical)*255  # found by Ka Ho to work
        kernel = np.ones((20, 20), np.uint8)
        ret, _ = cv2.threshold(empirical.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx((empirical > ret).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = remove_small_objects(mask.astype(bool), min_size=image.shape[0] * small_obj_size_factor)
        return mask.mean() > threshold and not sat_mean < sat_thresh

    @staticmethod
    def is_blurred(image):
        return False

    @staticmethod
    def is_folded(image):
        return False


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(usage="wsi_reader.py path_to_image [options]")
    parser.add_argument('--qc_mpp', default=1.0, type=float, help="MPP value to perform quality control on slide")
    parser.add_argument('--mpp', default=0.25, type=float, help="MPP value to read images from slide")
    parser.add_argument('--data_dir', type=str, default='', help="Dir where to save qc result")
    parser.add_argument('--check_tile_blur', action='store_true', help="Check for blur")
    parser.add_argument('--check_tile_fold', action='store_true', help="Check tile fold")
    parser.add_argument('--overwrite_qc', action='store_true', help="Overwrite saved quality control data")
    parser.add_argument('--patch_size', type=int, default=256, help="Pixel size of patches (at desired resolution)")
    parser.add_argument('--verbose', action='store_true', help="Print more information")
    opt, unknown = parser.parse_known_args()

    slide_path = Path(sys.argv[1])
    if not slide_path.is_file():
        raise ValueError(f"Invalid file input {str(slide_path)}")

    wsi_reader = WSIReader(opt, slide_path)
    wsi_reader.find_tissue_locations()
    wsi_reader.export_tissue_tiles('tiles_temp')








