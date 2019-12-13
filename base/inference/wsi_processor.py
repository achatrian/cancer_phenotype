from pathlib import Path
from tempfile import TemporaryFile
from functools import partial
import numpy as np
from tqdm import tqdm
from openslide import OpenSlideUnsupportedFormatError, OpenSlideError
import pyvips
from utils.utils import segmap2img
from data.images.wsi_reader import WSIReader


class WSIProcessor(WSIReader):

    def __init__(self, file_name, opt):
        file_name = str(file_name)
        super().__init__(file_name, opt)
        self.find_tissue_locations()  # find tissue locations where to apply function (H&E threshold isn't perfect but it seems to work for desired resolution)

    def apply(self, function, output_dtype, save_path):
        print(f"Applying function '{function.func.__name__ if isinstance(function, partial) else function.__name__}' on level {self.read_level} (target mpp: {self.opt.mpp})")
        # save output file
        prob_maps_dir = Path(save_path, 'prob_maps')
        prob_maps_dir.mkdir(exist_ok=True, parents=True)
        prob_map_path = prob_maps_dir/Path(self.file_name).with_suffix('.tiff').name
        print("[1/3] \t inferring probability map ...")
        try:
            prob_map = WSIReader(file_name=str(prob_map_path), opt=self.opt)
        except (OpenSlideUnsupportedFormatError, OpenSlideError):
            with TemporaryFile() as output_temporary_file:
                output = np.memmap(output_temporary_file, dtype=output_dtype, mode='w+',
                                   shape=tuple(self.level_dimensions[self.read_level])[::-1] + (self.opt.num_class,))
                for x, y in tqdm(self.tissue_locations):
                    input_tile = self.read_region((x, y), self.read_level, (self.opt.patch_size,)*2)
                    output_tile = function(input_tile)
                    output[y:y + self.opt.patch_size, x:x + self.opt.patch_size] = output_tile
                self.write_to_tiff(output, prob_map_path,  # TODO test this doesn't remove more than just extension
                                   tile=True, tile_width=512, tile_height=512, pyramid=True,
                                   bigtiff=output.size > 4294967295,
                                   compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
        # save shifted output file
        shifted_prob_maps_dir = Path(save_path, 'prob_maps_0.5shift')
        shifted_prob_maps_dir.mkdir(exist_ok=True, parents=True)
        shifted_prob_map_path = shifted_prob_maps_dir/Path(self.file_name).with_suffix('.tiff').name
        print("[2/3] \t inferring shifted probability map ...")
        try:
            shifted_prob_map = WSIReader(file_name=str(shifted_prob_map_path), opt=self.opt)
        except (OpenSlideUnsupportedFormatError, OpenSlideError):
            with TemporaryFile() as shifted_output_temporary_file:
                shifted_output = np.memmap(shifted_output_temporary_file, dtype=output_dtype, mode='w+',
                                           shape=tuple(self.level_dimensions[self.read_level])[::-1] + (self.opt.num_class,))
                xs, ys = tuple(x for x, y in self.tissue_locations), tuple(y for x, y in self.tissue_locations)
                xs, ys = (xs[0],) + tuple(min(int(x + 0.5 * self.opt.patch_size), self.level_dimensions[self.read_level][0] - self.opt.patch_size) for x in xs[:-1]), \
                         (ys[0],) + tuple(min(int(y + 0.5 * self.opt.patch_size), self.level_dimensions[self.read_level][1] - self.opt.patch_size) for y in ys[:-1])  # skip last entry as it's outside of image
                shifted_coords = tuple(zip(xs, ys))
                for x, y in tqdm(shifted_coords):
                    input_tile = self.read_region((x, y), self.read_level, (self.opt.patch_size,)*2)
                    output_tile = function(input_tile)
                    shifted_output[y:y + self.opt.patch_size, x:x + self.opt.patch_size] = output_tile
                self.write_to_tiff(shifted_output, shifted_prob_map_path,
                                   tile=True, tile_width=512, tile_height=512, pyramid=True,
                                   bigtiff=shifted_output.size > 4294967295,
                                   compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
        # load two files and overlap
        print("[3/3] \t making segmentation map ...")
        prob_map, shifted_prob_map = WSIReader(file_name=str(prob_map_path), opt=self.opt), \
                                     WSIReader(file_name=str(shifted_prob_map_path), opt=self.opt)
        merged_path = Path(save_path)/Path(self.file_name).with_suffix('.tiff').name
        with TemporaryFile() as merged_temporary_file:
            merged = np.memmap(merged_temporary_file, dtype=output_dtype, mode='w+',
                               shape=self.level_dimensions[self.read_level][::-1])  # FIXME must give output number of channels of segmentation map
            for x, y in tqdm(self.tissue_locations):
                input_tile = prob_map.read_region((x, y), self.read_level, (self.opt.patch_size,)*2)
                shifted_input_tile = shifted_prob_map.read_region((x, y), self.read_level, (self.opt.patch_size,)*2)
                merged[y:y + self.opt.patch_size, x:x + self.opt.patch_size] = self.merge_tiles(input_tile, shifted_input_tile)
            self.write_to_tiff(merged, merged_path,  # TODO test this doesn't remove more than just extension
                               tile=True, tile_width=512, tile_height=512, pyramid=True,
                               bigtiff=merged.size > 4294967295,
                               compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
        print("Done!")

    @staticmethod
    def merge_tiles(mask0, mask1, overlap=0.5, window=None):
        x, y = np.meshgrid(np.arange(mask0.shape[1]), np.arange(mask0.shape[0]))
        if window is None:
            def w(x, y):
                r"""Line increase from 0 to 1 at overlap end, plateau at 1 in center,
                and decay from overlap start to end of image"""
                lx, ly, r = (mask0.shape[1] - 1), (mask0.shape[0] - 1), overlap
                if 0 <= x < lx * r:
                    wx = x / (lx * r)
                elif lx * r <= x < lx * (1 - r):
                    wx = 1.0
                elif lx * (1 - r) <= x <= lx:
                    wx = -x / (lx * r) + 1 / r
                else:
                    raise ValueError(f"x must be in range [{0}, {lx}]")
                if 0 <= y < ly * r:
                    wy = y / (ly * r)
                elif ly * r <= y < ly * (1 - r):
                    wy = 1.0
                elif ly * (1 - r) <= y <= ly:
                    wy = -y / (ly * r) + 1 / r
                else:
                    raise ValueError(f"y must be in range [{0}, {ly}]")
                return wx * wy

            window = w
        window = np.vectorize(window)
        weights0 = np.tile(window(x, y)[..., np.newaxis], (1, 1, 3))
        weights1 = np.ones_like(mask0) - weights0  # complementary, so w0(x, y) + w1(x, y) = 1
        mean_mask = (weights0 * mask0 / 255.0 + weights1 * mask1 / 255.0) / 2
        image = segmap2img(mean_mask)
        return np.tile(image[..., np.newaxis], (1, 1, 3))

    @staticmethod
    def write_to_tiff(array, filename, **kwargs):
        # See https://jcupitt.github.io/libvips/API/current/VipsForeignSave.html#vips-tiffsave for tiffsave parameters
        dtype_to_format = {
            'uint8': 'uchar',
            'int8': 'char',
            'uint16': 'ushort',
            'int16': 'short',
            'uint32': 'uint',
            'int32': 'int',
            'float32': 'float',
            'float64': 'double',
            'complex64': 'complex',
            'complex128': 'dpcomplex',
        }
        height, width = array.shape[:2]
        bands = array.shape[2] if len(array.shape) == 3 else 1
        img = pyvips.Image.new_from_memory(array.ravel().data, width, height, bands, dtype_to_format[str(array.dtype)])
        img.tiffsave(str(filename), **kwargs)
