import os
import warnings
from openslide import OpenSlide
import numpy as np
import cv2
from skimage.morphology import remove_small_objects


class WSIReader(OpenSlide):

    def __init__(self, opt, file_name):
        super(WSIReader, self).__init__(file_name)
        slide_format = WSIReader.detect_format(file_name)
        if not slide_format:
            warnings.warn("Format vendor is not specified in metadata for {}".format(file_name), UserWarning)
        self.opt = opt
        self.file_name = file_name

    def find_good_locations(self):
        """
        Perform quality control on regions of slide by examining each tile
        :return:
        """
        good_locations = []
        # look at image with 1/16 the resolution of slide (alternative is to use get_thumbnail method for retrieving the whole slide)
        qc_level = 4 if self.level_count > 4 else self.level_count - 1  # qc stands for quality control
        qc_sizes = (self.opt.crop_size * (2 ** (qc_level - self.opt.wsi_read_level)),) * 2
        for x in range(0, self.level_dimensions[0][0],  qc_sizes[0]):  # dimensions = (width, height)
            for y in range(0, self.level_dimensions[0][1], qc_sizes[1]):
                tile = self.read_region((x, y), qc_level, qc_sizes)
                tile = np.array(tile)  # convert from PIL.Image
                if is_HnE(tile, self.level_dimensions[0]) and not (self.opt.check_tile_blur and is_blurred(tile)) and not (
                         self.opt.check_tile_fold and is_folded(tile)):
                    good_locations.append((x, y))
        self.good_locations = good_locations

    def __len__(self):
        return len(self.good_locations)

    def __getitem__(self, item):
        return self.read_region(self.good_locations[item], self.opt.wsi_read_level, (self.opt.crop_size, ) * 2)

    def get_locations_repr(self):
        return os.path.basename(self.file_name) + '\t' + '\t'.join(['{},{}'.format(x,y) for x,y in self.good_locations])


# functions for quality control on histology tiles
def is_HnE(image, size_wsi, threshold=0.5):
    """Returns true if slide contains tissue or just background"""
    # FIXME still not working properly - check all cases !
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





