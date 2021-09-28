from pathlib import Path
from tempfile import TemporaryFile
from functools import partial
from random import shuffle
import multiprocessing as mp
from os import remove
import json
import warnings
import numpy as np
from imageio import imread
from tqdm import tqdm
from skimage.transform import resize
from openslide import OpenSlideUnsupportedFormatError, OpenSlideError
import pyvips
from base.utils.utils import segmap2img
from data.images.wsi_reader import make_wsi_reader, add_reader_args, get_reader_options
from annotation.annotation_builder import AnnotationBuilder
from base.utils import debug


def read_tile(locations, queue_out, filename, opt):
    if not locations:
        warnings.warn("Worker has no locations to process")
        return
    slide = make_wsi_reader(filename, opt, opt.set_mpp)
    if opt.force_base_level_read:
        slide.read_level = 0
    read_level_patch_size = round(opt.patch_size * opt.mpp/(slide.mpp_x*slide.level_downsamples[slide.read_level]))
    for x, y in locations:
        try:
            tile = np.array(slide.read_region((x, y), slide.read_level, (read_level_patch_size,) * 2))[..., :3]
        except OpenSlideError:
            slide = make_wsi_reader(filename, opt, opt.set_mpp)
            if opt.force_base_level_read:
                slide.read_level = 0
            continue
        assert tile.shape[0] == read_level_patch_size and tile.shape[1] == read_level_patch_size
        if read_level_patch_size != opt.patch_size:
            tile = resize(tile, (opt.patch_size,)*2, preserve_range=True).astype(np.uint8)
        assert tile.shape[0] == opt.patch_size, f"tiles fed to network must be of size {opt.patch_size} and not {tile.shape[0]}"
        queue_out.put((x, y, tile))
    for i in range(opt.workers):
        queue_out.put(None)


def read_tiles(locations, queue_out, filename0, filename1, opt):
    r"""
    Given two slides, tiles are extracted from the level with resolution closest to desired resolution (opt.mpp)
    If the slides' resolution is different from the desired resolution, images are rescaled to obtain the desired
    resolution. The tiles are fed to an output queue
    :param locations:
    :param queue_out:
    :param filename0:
    :param filename1:
    :param opt:
    :return:
    """
    if not locations:
        warnings.warn("Worker has no locations to process")
        return
    slide0, slide1 = make_wsi_reader(filename0, opt, opt.set_mpp), make_wsi_reader(filename1, opt, opt.set_mpp)
    if opt.force_base_level_read:
        slide0.read_level, slide1.read_level = 0, 0
    # find read level patch_size for desired mpp
    read_level_patch_size = round(opt.patch_size * opt.mpp/(slide0.mpp_x*slide0.level_downsamples[slide0.read_level]))
    # assume both slides have the same mpp
    for x, y in locations:
        try:
            tile0 = np.array(slide0.read_region((x, y), slide0.read_level, (read_level_patch_size,) * 2))[..., :3]
            tile1 = np.array(slide1.read_region((x, y), slide1.read_level, (read_level_patch_size,) * 2))[..., :3]
        except OpenSlideError:  # some red tiles in images make script crash with error 'Cannot read raw tile'
            # OpenSlide has latching error semantics: once OpenSlideError is raised, all future operations on the
            # OpenSlide, other than close(), will also raise OpenSlideError.
            # --> need to reinitialized slides when error is encountered
            slide0, slide1 = make_wsi_reader(filename0, opt, opt.set_mpp), make_wsi_reader(filename1, opt, opt.set_mpp)
            if opt.force_base_level_read:
                slide0.read_level, slide1.read_level = 0, 0
            continue
        if read_level_patch_size != opt.patch_size:
            tile0 = resize(tile0, (opt.patch_size,)*2, preserve_range=True).astype(np.uint8)
            tile1 = resize(tile1, (opt.patch_size,)*2, preserve_range=True).astype(np.uint8)
        assert tile0.shape[0] == opt.patch_size, f"tiles fed to network must be of size {opt.patch_size} and not {tile0.shape[0]}"
        queue_out.put((x, y, tile0, tile1))
    for i in range(opt.workers):
        queue_out.put(None)


def get_locations_from_tissue_mask(tissue_mask, tissue_mask_patch_size, base_level_patch_size, slide, threshold=0.3):
    r"""Find locations in tissue mask where tissue was detected and map them to original slide"""
    if len(set(tissue_mask.flatten().tolist())) != 2:
        raise ValueError("Loaded image is not a binary tissue mask")
    values = np.unique(tissue_mask)
    tissue_mask[tissue_mask == values[0]] = 0
    tissue_mask[tissue_mask == values[1]] = 1
    tissue_mask = tissue_mask.astype(np.bool)
    tissue_locations = []  # locations with original slide dimensions where one can find tissue
    for j in range(slide.level_dimensions[0][1]//base_level_patch_size):
        for i in range(slide.level_dimensions[0][0]//base_level_patch_size):
            tissue_patch = tissue_mask[j*tissue_mask_patch_size:(j+1)*tissue_mask_patch_size,
                                       i*tissue_mask_patch_size:(i+1)*tissue_mask_patch_size]
            if tissue_patch.sum() > threshold*tissue_patch.size:
                tissue_locations.append((i*base_level_patch_size, j*base_level_patch_size))
    return tissue_locations


# TODO is this a god object - how could it be broken down?


class WSIProcessor:

    def __init__(self, file_name, opt, shift_and_merge=True, normalize_output=False, filter_location=None, set_mpp=None,
                 tissue_mask=None, tissue_mask_info=None, force_base_level_read=False):
        r"""
        Image wrapper for applying arbitrary segmentation or classification function to all tiles in an image
        :param file_name: path to slide
        :param opt: options (namespace returned by ProcessOpenSlide.parse())
        :param shift_and_merge: whether to interpolate two images tiled at intercalating locations to produce final image
        :param normalize_output: whether the function output needs to be normalised at the slide level before conversion to image
        :param filter_location:
        :param set_mpp: provides information on slide resolution if slide metadata does not contain it
        :param tissue_mask: tissue mask for slide
        :param tissue_mask_info: resolution info of tissue mask
        :param force_base_level_read: all tiles are read from level 0 and then rescaled - use if higher zoom levels have conversion artefacts
        apply() resizes tiles twice. The first time it reads the tile and resize it so that it matches the mpp the network expects.
        The second time it resizes the tiles so that they can be written into a tiff with the same base level as the read image.
        """
        # TODO read level is used to determine whether tile should be resized to fit tile size at base level corresponding
        # TODO to size on an image with lower resolution. Read level was initially conceived as being the level of the tiff
        # TODO that was most convenient to read from given the desired mpp, but since quality is higher when reading from bottom level
        # TODO this has changed. This has changed because of the force_base_level_read option
        # TODO The change in quality at higher level isn't really visible, unless one zooms into subpixel resolution.
        # TODO should an actual vs optimal read level introduced, so that one can estimate the optimal read level while giving choice over which level
        # TODO actually to read from, or should read level not used at all to determine resizing?
        # TODO resizing must happen twice
        # 28/02/21 disabling option as default because it might be creating problems with the segmentation, also I'm unsure it's required
        self.slide_id = Path(file_name).with_suffix('').name
        self.file_name = str(file_name)
        self.opt = opt
        # confusing -- patch size here refers to input tile size given to network
        # if network works on level different from base level, patch size for slide reader will be different
        self.slide = make_wsi_reader(self.file_name, opt, set_mpp)  # not actually used for reading regions in extract_tiles
        self.shift_and_merge = shift_and_merge
        self.normalize_output = normalize_output
        self.base_level_patch_size = round(self.opt.patch_size * self.slide.level_downsamples[self.slide.read_level])
        if tissue_mask is None:
            tile_read_errors = self.slide.find_tissue_locations(self.opt.tissue_threshold, self.opt.saturation_threshold)
            # find tissue locations where to apply function (H&E threshold isn't perfect but it seems to work for desired resolution)
            if tile_read_errors:  # if any instance of OpenSlideError is raised, slide properties become unreadable
                tissue_locations = self.slide.tissue_locations
                self.slide = make_wsi_reader(self.file_name, opt, set_mpp)
                self.slide.tissue_locations = tissue_locations
        else:
            self.tissue_mask = tissue_mask.astype(np.bool)
            self.tissue_mask_scaling, self.tissue_mask_patch_size = None, None
            if tissue_mask_info is None:
                raise ValueError("Tissue mask information is missing")
            self.tissue_mask_scaling = int(tissue_mask_info['level_downsamples'][tissue_mask_info['thumbnail_level']])
            self.tissue_mask_patch_size = self.base_level_patch_size//self.tissue_mask_scaling
            self.slide.tissue_locations = get_locations_from_tissue_mask(self.tissue_mask,
                                                                         self.tissue_mask_patch_size,
                                                                         self.base_level_patch_size,
                                                                         self.slide)
        self.filter_location = filter_location
        if self.filter_location is not None:
            self.slide.filter_locations([filter_location])
        if not self.slide.tissue_locations:
            raise ValueError(f"No image locations to process for slide {self.slide_id}")
        self.force_base_level_read = force_base_level_read
        self.opt.force_base_level_read = self.force_base_level_read

    def extract_tiles(self, locations, path0=None, path1=None):
        if not locations:
            raise ValueError("No specified locations for tile extraction")
        if self.opt.workers > 0:  # parallelize
            queue_out = mp.JoinableQueue(self.opt.batch_size*10)
            processes = []
            locations = list(locations)
            shuffle(locations)
            for i in range(self.opt.workers):
                if i != self.opt.workers - 1:
                    split = locations[i*len(locations)//self.opt.workers:(i+1)*len(locations)//self.opt.workers]
                else:
                    split = locations[i*len(locations)//self.opt.workers:]
                if path0 is not None and path1 is not None:
                    process = mp.Process(target=read_tiles, args=(split, queue_out, str(path0), str(path1), self.opt))
                else:
                    process = mp.Process(target=read_tile, args=(split, queue_out, self.file_name, self.opt))
                process.start()
                processes.append(process)
            n_active_processes = self.opt.workers
            while n_active_processes:
                data = queue_out.get(timeout=300)
                if data is not None:
                    yield data
                else:
                    n_active_processes -= 1
                queue_out.task_done()
            for process in processes:
                process.join(timeout=300)
        else:
            slide0 = make_wsi_reader(self.file_name, self.opt, getattr(self.opt, 'set_mpp', None)) if path0 is None \
                else make_wsi_reader(path0, self.opt, getattr(self.opt, 'set_mpp', None))
            slide1 = None if path1 is None else make_wsi_reader(path0, self.opt, getattr(self.opt, 'set_mpp', None))
            original_read_level = slide0.read_level
            if self.opt.force_base_level_read:
                slide0.read_level = 0
                if slide1:
                    slide1.read_level = 0
            read_level_patch_size = round(
                self.opt.patch_size * self.opt.mpp / (slide0.mpp_x * slide0.level_downsamples[slide0.read_level]))

            for x, y in locations:
                try:
                    tile0 = np.array(slide0.read_region((x, y), slide0.read_level, (read_level_patch_size,) * 2))[..., :3]
                    tile0 = resize(tile0, (self.opt.patch_size,)*2, preserve_range=True).astype(np.uint8)
                    if slide1 is not None:
                        tile1 = np.array(slide1.read_region((x, y), slide0.read_level, (read_level_patch_size,) * 2))[..., :3]
                        tile1 = resize(tile1, (self.opt.patch_size,) * 2, preserve_range=True).astype(np.uint8)
                        yield x, y, tile0, tile1
                    else:
                        yield x, y, tile0
                except OpenSlideError:  # 'Cannot read raw tile' error on red tiles in images
                    slide0 = self.slide if path0 is None else \
                        make_wsi_reader(path0, self.opt, getattr(self.opt, 'set_mpp', None))
                    slide1 = None if path1 is None else make_wsi_reader(path0, self.opt, getattr(self.opt, 'set_mpp', None))
                    if self.opt.force_base_level_read:
                        slide0.read_level = 0
                        if slide1:
                            slide1.read_level = 0
                            yield (None,)*4
                        yield (None,) * 3
                    continue
                slide0.read_level = original_read_level

    def apply(self, function, output_dtype, save_path):
        print(f"Applying function '{function.func.__name__ if isinstance(function, partial) else function.__name__}' on level {self.slide.read_level} (target mpp: {self.opt.mpp})")
        # save output file
        prob_maps_dir = Path(save_path, 'prob_maps') if self.shift_and_merge else Path(save_path)
        prob_maps_dir.mkdir(exist_ok=True, parents=True)
        prob_map_path = prob_maps_dir / Path(self.file_name).with_suffix('.tiff').name
        write_locations = [
            (x, y) for x, y in self.slide.tissue_locations
            if x < self.slide.level_dimensions[0][0] - self.base_level_patch_size and
               y < self.slide.level_dimensions[0][1] - self.base_level_patch_size
        ]  # don't process tiles near borders of image, so that all assignments are to a region of equal sides
        if not write_locations:
            warnings.warn(f"No tissue tiles in {self.slide_id} (tissue_threshold = {self.opt.tissue_threshold})")
            return False
        if (not hasattr(self.opt, 'set_mpp') or self.opt.set_mpp is None) and hasattr(self.slide, 'mpp_x'):
            self.opt.set_mpp = self.slide.mpp_x  # assign reference mpp to newly created tiffs for level selection
        if self.shift_and_merge:
            print("[1/3] \t inferring probability map ...")
        try:
            prob_map = make_wsi_reader(file_name=str(prob_map_path), opt=self.opt, set_mpp=self.opt.set_mpp)
            if self.opt.overwrite:
                prob_map.close()
                remove(prob_map_path)
                raise FileNotFoundError("overwriting prob map")
        except (OpenSlideUnsupportedFormatError, OpenSlideError, FileNotFoundError):
            with TemporaryFile() as output_temporary_file:
                output = np.memmap(output_temporary_file, dtype=output_dtype, mode='w+',
                                   shape=tuple(self.slide.level_dimensions[0])[::-1] + (3,))
                tiles = self.extract_tiles(write_locations)
                input_tiles, input_coordinates = [], []  # store tiles until desired batch size is reached
                failed_reads = 0
                for x, y, input_tile in tqdm(tiles, total=len(write_locations)):
                    if x is None:
                        failed_reads += 1
                        continue
                    input_tiles.append(input_tile), input_coordinates.append((x, y))
                    if len(input_tiles) == self.opt.batch_size or len(input_coordinates) == len(write_locations):
                        output_tiles = function(input_tiles if len(input_tiles) > 1 else input_tiles[0])  # sequence of tiles
                        if not isinstance(output_tiles, (list, tuple)):
                            output_tiles = [output_tiles]  # one tile case
                        for output_tile, (x_, y_) in zip(output_tiles, input_coordinates):
                            if self.slide.read_level != 0:
                                output_tile = resize(output_tile, (self.base_level_patch_size,)*2,
                                                      preserve_range=True).astype(np.uint8)
                            if self.tissue_mask is not None:
                                x_patch, y_patch = x_//self.tissue_mask_scaling, y_//self.tissue_mask_scaling
                                tissue_patch = self.tissue_mask[y_patch:self.tissue_mask_patch_size+y_patch,
                                                                x_patch:self.tissue_mask_patch_size+x_patch]
                                tissue_patch = resize(tissue_patch, (self.base_level_patch_size,)*2, 0)[..., np.newaxis]
                                output_tile = output_tile * tissue_patch
                            output[y_:y_ + self.base_level_patch_size, x_:x_ + self.base_level_patch_size] = output_tile
                        input_tiles.clear(), input_coordinates.clear()
                if failed_reads:
                    print(f"{failed_reads} tiles could not be read (slide dim: {self.slide.level_dimensions[0]}, "
                        f"basal patch size: {self.base_level_patch_size})")
                # if output is not an RGB image, normalize it
                output_max, output_min = output.max(), output.min()
                if self.normalize_output:
                    output = self.normalize_to_rgb(output, output_max, output_min)
                print(f"Writing tiff to {prob_map_path} ...")
                self.write_to_tiff(output, prob_map_path,  # TODO test this doesn't remove more than just extension
                                   tile=True, tile_width=512, tile_height=512, pyramid=True,
                                   bigtiff=output.size > 4294967295,
                                   compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
        if self.shift_and_merge:
            # save shifted output file
            shifted_prob_maps_dir = Path(save_path, 'prob_maps_0.5shift')
            shifted_prob_maps_dir.mkdir(exist_ok=True, parents=True)
            shifted_prob_map_path = shifted_prob_maps_dir / Path(self.file_name).with_suffix('.tiff').name
            print("[2/3] \t inferring shifted probability map ...")
            try:
                shifted_prob_map = make_wsi_reader(file_name=str(shifted_prob_map_path), opt=self.opt, set_mpp=self.opt.set_mpp)
                if self.opt.overwrite:
                    shifted_prob_map.close()
                    remove(shifted_prob_map_path)
                    raise FileNotFoundError("overwriting shifted prob map")
            except (OpenSlideUnsupportedFormatError, OpenSlideError, FileNotFoundError):
                with TemporaryFile() as shifted_output_temporary_file:
                    shifted_output = np.memmap(shifted_output_temporary_file, dtype=output_dtype, mode='w+',
                                               shape=tuple(self.slide.level_dimensions[0])[::-1] + (3,))
                    xs, ys = tuple(x for x, y in write_locations), tuple(y for x, y in write_locations)
                    xs, ys = (xs[0],) + tuple(min(round(x + 0.5 * self.base_level_patch_size),
                                                  self.slide.level_dimensions[0][
                                                      0] - self.base_level_patch_size) for x in xs[:-1]), \
                             (ys[0],) + tuple(min(round(y + 0.5 * self.base_level_patch_size),
                                                  self.slide.level_dimensions[0][
                                                      1] - self.base_level_patch_size) for y in ys[:-1])
                    # skip last entry as it's outside of image
                    shifted_coords = tuple(zip(xs, ys))
                    shifted_tiles = self.extract_tiles(shifted_coords)
                    input_tiles, input_coordinates = [], []
                    for x, y, input_tile in tqdm(shifted_tiles, total=len(shifted_coords)):
                        input_tiles.append(input_tile), input_coordinates.append((x, y))
                        if len(input_tiles) == self.opt.batch_size or len(input_coordinates) == len(shifted_coords):
                            output_tiles = function(input_tiles if len(input_tiles) > 1 else input_tiles[0])  # sequence of tiles
                            if not isinstance(output_tiles, (list, tuple)):
                                output_tiles = [output_tiles]  # one tile case
                            for output_tile, (x_, y_) in zip(output_tiles, input_coordinates):
                                if self.slide.read_level != 0:
                                    output_tile = resize(output_tile, (self.base_level_patch_size,)*2,
                                                          preserve_range=True).astype(np.uint8)
                                if self.tissue_mask is not None:
                                    x_patch, y_patch = x_ // self.tissue_mask_scaling, y // self.tissue_mask_scaling
                                    tissue_patch = self.tissue_mask[y_patch:self.tissue_mask_patch_size + y_patch,
                                                                    x_patch:self.tissue_mask_patch_size + x_patch]
                                    tissue_patch = resize(tissue_patch, (self.base_level_patch_size,)*2, 0)[..., np.newaxis]
                                    output_tile = output_tile * tissue_patch
                                shifted_output[y_:y_ + self.base_level_patch_size, x_:x_ + self.base_level_patch_size] = output_tile
                            input_tiles.clear(), input_coordinates.clear()
                    print(f"Writing tiff to {shifted_prob_map_path} ...")
                    self.write_to_tiff(shifted_output, shifted_prob_map_path,
                                       tile=True, tile_width=512, tile_height=512, pyramid=True,
                                       bigtiff=shifted_output.size > 4294967295,
                                       compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
            # load two files and overlap
            print("[3/3] \t making segmentation map ...")
            merged_path = Path(save_path) / Path(self.file_name).with_suffix('.tiff').name
            try:
                merged = make_wsi_reader(file_name=str(merged_path), opt=self.opt, set_mpp=self.opt.set_mpp)
                if self.opt.overwrite:
                    merged.close()
                    raise FileNotFoundError("overwriting merged")
            except FileNotFoundError:
                with TemporaryFile() as merged_temporary_file:
                    merged = np.memmap(merged_temporary_file, dtype=output_dtype, mode='w+',
                                       shape=tuple(self.slide.level_dimensions[0])[::-1] + (3,))  # FIXME must give output number of channels of segmentation map
                    offset_tiles = self.extract_tiles(write_locations, str(prob_map_path), str(shifted_prob_map_path))
                    for x, y, input_tile, shifted_input_tile in tqdm(offset_tiles, total=len(write_locations)):
                        merged_tile = self.merge_tiles(input_tile, shifted_input_tile)
                        if self.slide.read_level != 0:
                            merged_tile = resize(merged_tile, (self.base_level_patch_size,) * 2,
                                                 preserve_range=True).astype(np.uint8)
                        merged[y:y + self.base_level_patch_size, x:x + self.base_level_patch_size] = merged_tile
                    print(f"Writing tiff to {merged_path} ...")
                    self.write_to_tiff(merged, merged_path,  # TODO test this doesn't remove more than just extension
                                       tile=True, tile_width=512, tile_height=512, pyramid=True,
                                       bigtiff=merged.size > 4294967295,
                                       compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
        print("Done!")
        return True

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

    @staticmethod
    def normalize_to_rgb(memmap, max_=None, min_=None):
        if max_ is None:
            max_ = memmap.max()
        if min_ is None:
            min_ = memmap.min()
        if min_ < 0:
            memmap = np.add(memmap, abs(min_), out=memmap)  # shift to make all values positive
            max_ = max_ + abs(min_)
        memmap = np.true_divide(memmap, max_, out=memmap)
        memmap = np.multiply(memmap, 255, out=memmap)
        if memmap.dtype == np.float16:
            output_view_uint = memmap.view(np.uint16)
            output_view_uint[:] = memmap.astype(np.uint16)
        elif memmap.dtype == np.float32:
            output_view_uint = memmap.view(np.uint32)
            output_view_uint[:] = memmap.astype(np.uint32)
        elif memmap.dtype == np.float64:
            output_view_uint = memmap.view(np.uint64)
            output_view_uint[:] = memmap.astype(np.uint64)
        else:
            raise TypeError(f"Invalid input dtype '{memmap.dtype}'")
        strides = output_view_uint.strides
        shape = output_view_uint.shape
        output_view_uint8 = output_view_uint.view(np.uint8)
        output_view_uint8 = np.lib.stride_tricks.as_strided(output_view_uint8, shape=shape, strides=strides)  # THAT'S COOL!
        return output_view_uint8

    def apply_classification(self, function, classes, slide_label, save_dir):
        print(f"Applying function '{function.func.__name__ if isinstance(function, partial) else function.__name__}' on level {self.slide.read_level} (target mpp: {self.opt.mpp})")
        if isinstance(classes, int):
            classes = tuple(str(i) for i in range(classes))  # one layer per class
        process_locations = [
            (x, y) for x, y in self.slide.tissue_locations
            if x < self.slide.level_dimensions[0][0] - self.base_level_patch_size and
               y < self.slide.level_dimensions[0][1] - self.base_level_patch_size
        ]
        if not process_locations:
            warnings.warn(f"No tissue tiles in {self.slide_id} (tissue_threshold = {self.opt.tissue_threshold})")
            return False
        classification_results = []
        tiles = self.extract_tiles(process_locations)
        input_tiles, input_coordinates = [], []
        slide_id = Path(self.file_name).with_suffix('').name
        color_annotation = AnnotationBuilder(slide_id, 'classification', classes)
        colors = [{
                      'fill': {
                          'saturation': 1.0,
                          'lightness': 0.72,
                          'hue': i*(360//len(classes)),
                          'alpha': 1.0,
                      },
                      'stroke': {
                          'saturation': 1.0,
                          'lightness': 0.72,
                          'hue': i*(360//len(classes)),
                          'alpha': 1.0
                      }
                                              } for i in range(len(classes))]
        colorings_dir = save_dir / 'colorings'
        colorings_dir.mkdir(exist_ok=True, parents=True)
        failed_reads = 0
        for x, y, input_tile in tqdm(tiles, total=len(process_locations)):
            if x is None:
                failed_reads += 1
                continue
            input_tiles.append(input_tile), input_coordinates.append((x, y))
            if len(input_tiles) == self.opt.batch_size or len(input_coordinates) == len(process_locations):
                data = function(input_tiles if len(input_tiles) > 1 else input_tiles[0], slide_label)
                assert len(data['outputs']) == len(input_coordinates)
                for i, ((x_, y_), output) in enumerate(zip(input_coordinates, data['outputs'])):
                    classification = np.argmax(output)
                    probabilities = output.squeeze().tolist()
                    classification_results.append({
                        'x': x_, 'y': y_,
                        'classification': int(classification),
                    })
                    if 'variance' in data:
                        classification_results[-1].update(
                            variance=data['variance'][i],
                            loss_variance=data['loss_variance'][i]
                        )
                    if 'features' in data:
                        classification_results[-1].update(features=data['features'][i].tolist())
                    classification_results[-1].update({f'probability_class{i}': probability
                                                       for i, probability in enumerate(probabilities)})
                    color_annotation.add_item(classes[classification], 'rectangle', classes[classification],
                                              filled=True, color=colors[classification],
                                              rectangle=(x_, y_, self.base_level_patch_size, self.base_level_patch_size))
                input_tiles.clear(), input_coordinates.clear()
        if failed_reads:
            print(f"{failed_reads} tiles could not be read (slide dim: {self.slide.level_dimensions[0]}, "
                  f"basal patch size: {self.base_level_patch_size})")
        color_annotation.dump_to_json(colorings_dir)
        with open(save_dir/(slide_id + '.json'), 'w') as results_file:
            json.dump(classification_results, results_file)
        print("Done!")
