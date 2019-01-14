import os
import re
import warnings
import csv
from openslide import OpenSlide
import numpy as np
import cv2
from skimage.morphology import remove_small_objects
from base.utils import utils


class WSIReader(OpenSlide):

    def __init__(self, opt, file_name):
        super(WSIReader, self).__init__(file_name)
        slide_format = WSIReader.detect_format(file_name)
        if not slide_format:
            warnings.warn("Format vendor is not specified in metadata for {}".format(file_name), UserWarning)
        self.opt = opt
        self.file_name = file_name
        self.good_locations = []
        self.locations = []
        self.tile_info = dict()
        self.qc = None

    def save_locations(self):
        name_ext = re.sub('\.(ndpi|svs)', '', os.path.basename(self.file_name)) + '.tsv'  # strip extension and add .tsv
        slide_loc_path = os.path.join(self.opt.data_dir, 'data', 'tile_locations', name_ext)
        utils.mkdirs(os.path.join(self.opt.data_dir, 'data', 'tile_locations'))
        with open(slide_loc_path, 'w') as slide_loc_file:
            writer = csv.writer(slide_loc_file, delimiter='\t')
            for x, y in self.locations:
                loc = f'{x},{y}'
                writer.writerow((loc, self.tile_info[loc]))

    def read_locations(self):
        name_ext = re.sub('\.(ndpi|svs)', '', os.path.basename(self.file_name)) + '.tsv'  # strip extension and add .tsv
        slide_loc_path = os.path.join(self.opt.data_dir, 'data', 'tile_locations', name_ext)
        with open(slide_loc_path, 'w') as slide_loc_file:
            if os.stat(slide_loc_path).st_size == 0:  # rewrite if file is empty
                raise FileNotFoundError(f"Empty file: {slide_loc_file}")
            reader = csv.reader(slide_loc_file, delimiter='\t')
            for loc, info in reader:
                self.tile_info[loc] = info
                loc = tuple(int(d) for d in loc.split(','))
                self.locations.append(loc)
                if info == 'good':
                    self.good_locations.append(loc)

    def find_good_locations(self):
        """
        Perform quality control on regions of slide by examining each tile
        :return:
        """
        try:
            self.read_locations()
            if self.opt.verbose:
                print("Read qc info for {}".format(os.path.basename(self.file_name)))
        except FileNotFoundError:
            good_locations = []
            # scan image with 1/16 the resolution of slide (alternative is to use get_thumbnail method for retrieving the whole slide)
            qc_level = 3 if self.level_count > 3 else self.level_count - 1  # qc stands for quality control
            qc_sizes = (self.opt.crop_size * (2 ** (qc_level - self.opt.wsi_read_level)),) * 2
            qc_stride = self.opt.crop_size * (2 ** qc_level)
            for x in range(0, self.level_dimensions[0][0], qc_stride):  # dimensions = (width, height)
                for y in range(0, self.level_dimensions[0][1], qc_stride):
                    tile = self.read_region((x, y), qc_level, qc_sizes)
                    tile = np.array(tile)  # convert from PIL.Image
                    self.locations.append((x, y))
                    if not is_HnE(tile, self.level_dimensions[0]):
                        self.tile_info[f'{x},{y}'] = 'empty'
                    elif self.opt.check_tile_blur and is_blurred(tile):
                        self.tile_info[f'{x},{y}'] = 'blur'
                    elif self.opt.check_tile_fold and is_folded(tile):
                        self.tile_info[f'{x},{y}'] = 'folded'
                    else:
                        self.tile_info[f'{x},{y}'] = 'good'
                        good_locations.append((x, y))
            self.good_locations = good_locations
            self.save_locations()  # overwrite if existing
            if self.opt.verbose:
                print("Perform quality control on {}".format(os.path.basename(self.file_name)))

    def __len__(self):
        return len(self.good_locations)

    def __getitem__(self, item):
        return self.read_region(self.good_locations[item], self.opt.wsi_read_level, (self.opt.crop_size, ) * 2)

    def __repr__(self):
        # TODO what should this return
        return os.path.basename(self.file_name) + '\t' + '\t'.join(['{},{}'.format(x, y) for x, y in self.good_locations])


# functions for quality control on histology tiles
def is_HnE(image, size_wsi, threshold=0.5):
    """Returns true if slide contains tissue or just background"""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(float)
    empirical = hsv_image.prod(axis=2)  # found by Ka Ho to work
    empirical = empirical/np.max(empirical)*255  # found by Ka Ho to work
    kernel = np.ones((20, 20), np.uint8)
    ret, _ = cv2.threshold(empirical.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx((empirical > ret).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = remove_small_objects(mask.astype(bool), min_size=size_wsi[0]//1000)
    return mask.mean() > threshold  # if more than 50% (default) of pixels are H&E, return True


def is_blurred(image):
    return False


def is_folded(image):
    return False





