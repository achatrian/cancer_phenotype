from pathlib import Path
from tempfile import TemporaryFile
from functools import partial
from random import shuffle
import multiprocessing as mp
import json
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from openslide import OpenSlideUnsupportedFormatError, OpenSlideError
import pyvips
from base.utils.utils import segmap2img
from data.images.wsi_reader import WSIReader
from annotation.annotation_builder import AnnotationBuilder
from base.utils import debug


def read_tile(locations, queue_out, filename, opt):
    if not locations:
        raise ValueError("Worker has no locations to process")
    slide = WSIReader(filename, opt)
    read_level_patch_size = int(opt.patch_size * opt.mpp/(slide.mpp_x*slide.level_downsamples[slide.read_level]))
    for x, y in locations:
        tile = np.array(slide.read_region((x, y), slide.read_level, (read_level_patch_size,) * 2))
        assert tile.shape[0] == read_level_patch_size and tile.shape[1] == read_level_patch_size
        if read_level_patch_size != opt.patch_size:
            tile = resize(tile, (opt.patch_size,)*2,
                          preserve_range=True).astype(np.uint8)
        assert tile.shape[0] == opt.patch_size, f"tiles fed to network must be of size {opt.patch_size} and not {tile.shape[0]}"
        queue_out.put((x, y, tile))
    for i in range(opt.workers):
        queue_out.put(None)


class WSIProcessor:

    def __init__(self, file_name, opt, shift_and_merge=True, normalize_output=False, filter_location=None):
        self.file_name = str(file_name)
        self.opt = opt
        # confusing -- patch size here refers to input tile size given to network
        # if network works on level different from base level, patch size for slide reader will be different
        self.slide = WSIReader(self.file_name, opt)
        self.shift_and_merge = shift_and_merge
        self.normalize_output = normalize_output
        self.slide.find_tissue_locations(self.opt.tissue_threshold)  # find tissue locations where to apply function (H&E threshold isn't perfect but it seems to work for desired resolution)
        self.filter_location = filter_location
        if self.filter_location is not None:
            self.slide.filter_locations([filter_location])

    def extract_tiles(self, locations):
        if not locations:
            raise ValueError("No specified locations for tile extraction")
        if self.opt.workers > 0:  # parallelise
            queue_out = mp.JoinableQueue(self.opt.batch_size*10)
            processes = []
            locations = list(locations)
            shuffle(locations)
            for i in range(self.opt.workers):
                if i != self.opt.workers - 1:
                    split = locations[i*len(locations)//self.opt.workers:(i+1)*len(locations)//self.opt.workers]
                else:
                    split = locations[i*len(locations)//self.opt.workers:]
                process = mp.Process(target=read_tile,
                                     args=(split, queue_out, self.file_name, self.opt))
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
            read_level_patch_size = int(self.opt.patch_size * self.opt.mpp /
                                        (self.slide.mpp_x * self.slide.level_downsamples[self.slide.read_level]))
            for x, y in locations:
                tile = np.array(self.slide.read_region((x, y), 0, (read_level_patch_size,)*2))
                assert tile.shape[0] == read_level_patch_size
                tile = resize(tile, (self.opt.patch_size,)*2,
                          preserve_range=True).astype(np.uint8)
                yield x, y, tile

    def apply(self, function, output_dtype, save_path):
        print(f"Applying function '{function.func.__name__ if isinstance(function, partial) else function.__name__}' on level {self.slide.read_level} (target mpp: {self.opt.mpp})")
        # save output file
        prob_maps_dir = Path(save_path, 'prob_maps') if self.shift_and_merge else Path(save_path)
        prob_maps_dir.mkdir(exist_ok=True, parents=True)
        prob_map_path = prob_maps_dir / Path(self.file_name).with_suffix('.tiff').name
        base_level_patch_size = int(self.opt.patch_size*self.slide.level_downsamples[self.slide.read_level])
        write_locations = [
            (x, y) for x, y in self.slide.tissue_locations
            if x < self.slide.level_dimensions[0][0] - base_level_patch_size and
               y < self.slide.level_dimensions[0][1] - base_level_patch_size
        ]  # don't process tiles near borders of image, so that all assignments are to a region of equal sides
        if self.shift_and_merge:
            print("[1/3] \t inferring probability map ...")
        try:
            prob_map = WSIReader(file_name=str(prob_map_path), opt=self.opt)
        except (OpenSlideUnsupportedFormatError, OpenSlideError):
            with TemporaryFile() as output_temporary_file:
                output = np.memmap(output_temporary_file, dtype=output_dtype, mode='w+',
                                   shape=tuple(self.slide.level_dimensions[0])[::-1] + (3,))
                tiles = self.extract_tiles(write_locations)
                input_tiles, input_coordinates = [], []  # store tiles until desired batch size is reached
                for x, y, input_tile in tqdm(tiles, total=len(write_locations)):
                    input_tiles.append(input_tile), input_coordinates.append((x, y))
                    if len(input_tiles) == self.opt.batch_size or len(input_coordinates) == len(write_locations):
                        output_tiles = function(input_tiles if len(input_tiles) > 1 else input_tiles[0])  # sequence of tiles
                        if not isinstance(output_tiles, (list, tuple)):
                            output_tiles = [output_tiles]  # one tile case
                        for output_tile, (x_, y_) in zip(output_tiles, input_coordinates):
                            if self.slide.read_level != 0:
                                output_tile = resize(output_tile, (base_level_patch_size,)*2,
                                                      preserve_range=True).astype(np.uint8)
                            output[y_:y_ + base_level_patch_size, x_:x_ + base_level_patch_size] = output_tile
                        input_tiles.clear(), input_coordinates.clear()
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
                shifted_prob_map = WSIReader(file_name=str(shifted_prob_map_path), opt=self.opt)
            except (OpenSlideUnsupportedFormatError, OpenSlideError):
                with TemporaryFile() as shifted_output_temporary_file:
                    shifted_output = np.memmap(shifted_output_temporary_file, dtype=output_dtype, mode='w+',
                                               shape=tuple(self.slide.level_dimensions[0])[::-1] + (3,))
                    xs, ys = tuple(x for x, y in write_locations), tuple(
                        y for x, y in write_locations)
                    xs, ys = (xs[0],) + tuple(min(int(x + 0.5 * base_level_patch_size),
                                                  self.slide.level_dimensions[0][
                                                      0] - base_level_patch_size) for x in xs[:-1]), \
                             (ys[0],) + tuple(min(int(y + 0.5 * base_level_patch_size),
                                                  self.slide.level_dimensions[0][
                                                      1] - base_level_patch_size) for y in ys[:-1])
                    # skip last entry as it's outside of image
                    shifted_coords = tuple(zip(xs, ys))
                    shifted_tiles = self.extract_tiles(shifted_coords)
                    input_tiles, input_coordinates = [], []
                    for x, y, input_tile in tqdm(shifted_tiles, total=len(shifted_coords)):
                        input_tiles.append(input_tile), input_coordinates.append((x, y))
                        if len(input_tiles) == self.opt.batch_size or len(input_coordinates) == len(shifted_coords):
                            output_tiles = function(
                                input_tiles if len(input_tiles) > 1 else input_tiles[0])  # sequence of tiles
                            if not isinstance(output_tiles, (list, tuple)):
                                output_tiles = [output_tiles]  # one tile case
                            for output_tile, (x_, y_) in zip(output_tiles, input_coordinates):
                                if self.slide.read_level != 0:
                                    output_tile = resize(output_tile, (base_level_patch_size,)*2,
                                                          preserve_range=True).astype(np.uint8)
                                shifted_output[y_:y_ + base_level_patch_size, x_:x_ + base_level_patch_size] = output_tile
                            input_tiles.clear(), input_coordinates.clear()
                    print(f"Writing tiff to {shifted_prob_map_path} ...")
                    self.write_to_tiff(shifted_output, shifted_prob_map_path,
                                       tile=True, tile_width=512, tile_height=512, pyramid=True,
                                       bigtiff=shifted_output.size > 4294967295,
                                       compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
            # load two files and overlap
            print("[3/3] \t making segmentation map ...")
            prob_map, shifted_prob_map = WSIReader(file_name=str(prob_map_path), opt=self.opt), \
                                         WSIReader(file_name=str(shifted_prob_map_path), opt=self.opt)
            merged_path = Path(save_path) / Path(self.file_name).with_suffix('.tiff').name
            with TemporaryFile() as merged_temporary_file:
                merged = np.memmap(merged_temporary_file, dtype=output_dtype, mode='w+',
                                   shape=tuple(self.slide.level_dimensions[0])[::-1] + (3,))  # FIXME must give output number of channels of segmentation map
                for x, y in tqdm(write_locations):
                    # pyvips saves images with alpha channel
                    input_tile = np.array(prob_map.read_region((x, y), 0, (base_level_patch_size,) * 2))[..., :3]
                    shifted_input_tile = np.array(shifted_prob_map.read_region((x, y), 0, (base_level_patch_size,) * 2))[..., :3]
                    merged_tile = self.merge_tiles(input_tile, shifted_input_tile)
                    merged[y:y + base_level_patch_size, x:x + base_level_patch_size] = merged_tile
                print(f"Writing tiff to {merged_path} ...")
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

    def apply_classification(self, function, classes, save_dir):
        print(f"Applying function '{function.func.__name__ if isinstance(function, partial) else function.__name__}' on level {self.slide.read_level} (target mpp: {self.opt.mpp})")
        if isinstance(classes, int):
            classes = tuple(str(i) for i in range(classes))  # one layer per class
        base_level_patch_size = int(self.opt.patch_size * self.slide.level_downsamples[self.slide.read_level])
        process_locations = [
            (x, y) for x, y in self.slide.tissue_locations
            if x < self.slide.level_dimensions[0][0] - base_level_patch_size and
               y < self.slide.level_dimensions[0][1] - base_level_patch_size
        ]
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
                          'alpha': 0.7,
                      },
                      'stroke': {
                          'saturation': 1.0,
                          'lightness': 0.72,
                          'hue': i*(360//len(classes)),
                          'alpha': 0.0
                      }
                                              } for i in range(len(classes))]
        colorings_dir = save_dir / 'colorings'
        colorings_dir.mkdir(exist_ok=True, parents=True)
        for x, y, input_tile in tqdm(tiles, total=len(process_locations)):
            input_tiles.append(input_tile), input_coordinates.append((x, y))
            if len(input_tiles) == self.opt.batch_size or len(input_coordinates) == len(process_locations):
                outputs, losses = function(input_tiles if len(input_tiles) > 1 else input_tiles[0])
                assert len(outputs) == len(input_coordinates)
                for (x_, y_), output, loss in zip(input_coordinates, outputs, losses):
                    classification_results.append(((x_, y_), output, loss))
                    color_annotation.add_item(classes[output], 'rectangle', classes[output],
                                              filled=True, color=colors[output],
                                              rectangle=(x_, y_, base_level_patch_size, base_level_patch_size))
        color_annotation.dump_to_json(colorings_dir)
        with open(save_dir/(slide_id + '.json')) as results_file:
            json.dump(classification_results, results_file)







