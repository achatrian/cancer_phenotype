import warnings
from openslide import OpenSlide
from deploy import is_HnE, is_blurred, is_folded


class WSIReader(OpenSlide):

    def __init__(self, opt, file_name):
        super(WSIReader, self).__init__(file_name)
        slide_format = WSIReader.detect_format(file_name)
        if not slide_format:
            warnings.warn("Format vendor is not specified in metadata for {}".format(file_name), UserWarning)
        self.opt = opt
        self.file_name = file_name

    def find_good_locations(self):
        good_locations = []
        low_res_dimensions = self.level_dimensions[self.level_count - 1]
        low_res_crop_size = self.opt.crop_size / (2 ** (self.level_count - 1 - self.opt.wsi_read_level))
        for x in range(0, low_res_dimensions[0], self.opt.crop_size):  # dimensions = (width, height)
            for y in range(0, low_res_dimensions[1], self.opt.crop_size):
                region_size = (min(low_res_crop_size, low_res_dimensions[0] - x),
                               min(low_res_crop_size, low_res_dimensions[1] - y))  # for corner regions
                tile = self.read_region((x, y), self.level_count - 1, region_size)
                if is_HnE(tile) and (not self.opt.check_blur and is_blurred(tile)) and (
                        not self.opt.check_folds and is_folded(tile)):
                    good_locations.append((x, y))
        self.low_res_locations = good_locations
        coords_scale = 2 ** (self.level_count - 1 - self.opt.wsi_read_level)  # every level up is a 2x magnification
        self.high_res_locations = [(x * coords_scale, y * coords_scale) for x, y in self.low_res_locations]

    def __len__(self):
        return len(self.low_res_locations)

    def __getitem__(self, item):
        return self.read_region(self.high_res_locations[item], self.opt.wsi_read_level, self.opt.crop_size)




