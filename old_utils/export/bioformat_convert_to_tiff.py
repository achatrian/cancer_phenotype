# file where to test things
import argparse
from pathlib import Path
from itertools import product
import json
from tempfile import TemporaryFile
from random import shuffle
import numpy as np
import javabridge
import bioformats
import xmltodict
from tifffile import imwrite
#import pyvips
import tqdm

r"""Convert bioformats-compatible images into (big)tiff images. RUNS ONLY ON PYTHON <=3.5"""
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

if __name__ == '__main__':
    javabridge.start_vm(class_path=bioformats.JARS)
    ### programme starts here ###
    formats = ['.czi', '.svs', '.ndpi']
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=Path, required=True)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--source_extension', type=str, default='.czi', choices=formats)
    parser.add_argument('--target_dir', type=Path, default=None)
    parser.add_argument('--compress_level', type=int, default=6, help="adobe deflate compression level")
    parser.add_argument('--shuffle_images', action='store_true')
    args = parser.parse_args()
    if args.target_dir is not None:
        args.target_dir.mkdir(exist_ok=True, parents=True)
    image_paths = list(args.data_dir.glob('*{}'.format(args.source_extension))) + \
                  list(args.data_dir.glob('*/*{}'.format(args.source_extension)))
    shuffle(image_paths)
    if not args.data_dir.exists():
        raise ValueError("Specified data directory does not exist: {}".format(args.data_dir))
    print("Source dir: {}; target dir {}".format(args.data_dir, args.target_dir))
    if not image_paths:
        print("No images to process with extension {}".format(args.source_extension))
    for i, image_path in enumerate(image_paths):
        save_path = args.target_dir / image_path.with_suffix('.tiff').name \
            if args.target_dir is not None else image_path.with_suffix('.tiff')
        if save_path.exists() and not args.overwrite:
            continue
        print("Converting {} ...".format(image_path.name))
        md_seq = bioformats.get_omexml_metadata(path=str(image_path))
        md_dir = args.target_dir/'metadata'
        if not md_dir.is_dir():
            md_dir.mkdir(exist_ok=True)
        metadata = xmltodict.parse(md_seq)
        with (md_dir / image_path.with_suffix('.json').name).open(mode='w') as md_file:
            json.dump(metadata, md_file)
        level0 = metadata['OME']['Image'][0]['Pixels']  # access pixel info for level 0
        width, height = int(level0['@SizeX']), int(level0['@SizeY'])
        with TemporaryFile() as temporary:
            tiff_store = np.memmap(temporary, dtype=np.uint8, mode='w+', shape=(height, width, 3))
            # tiff_store = np.zeros((height, width, 3), dtype=np.uint8)  # pixel encoding needs to be <32bit for tiff
            xs, ys = list(range(0, width, args.tile_size)), list(range(0, height, args.tile_size))
            with bioformats.ImageReader(str(image_path)) as reader:
                for x, y in tqdm.tqdm(product(xs, ys), total=len(xs) * len(ys)):
                    if x == xs[-1]:
                        x = width - x
                    if y == ys[-1]:
                        y = height - y
                    tile = reader.read(XYWH=(x, y, args.tile_size, args.tile_size))
                    tile = tile * 255  # RGB is from 0 to 1 in .czi files
                    tiff_store[y:y + args.tile_size, x:x + args.tile_size] = tile.astype(np.uint8)
            imwrite(str(save_path), tiff_store, bigtiff=tiff_store.size > 4294967295, compress=args.compress_level)
            # PYVIPS requires python>=3.6, but JAVABRIDGE only runs on python<=3.5
            # img = pyvips.Image.new_from_memory(
            #     tiff_store.ravel().data,
            #     tiff_store.shape[0], tiff_store.shape[1], tiff_store.shape[2],
            #     dtype_to_format[str(tiff_store.dtype)]
            # )
            # # for files larger than 4GB, save as bigtiff
            # img.tiffsave(str(save_path), tile=True, pyramid=True, bigtiff=tiff_store.size > 4294967295,
            #              compression='VIPS_FOREIGN_TIFF_COMPRESSION_DEFLATE')
            del tiff_store  # free up memory?
        print("Saved images #{} at {}".format(i, str(save_path)))
    ### programme ends here
    javabridge.kill_vm()
