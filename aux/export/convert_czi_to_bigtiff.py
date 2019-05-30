# file where to test things
import argparse
from pathlib import Path
from itertools import product
import numpy as np
import javabridge
import bioformats
import xmltodict
from tifffile import imwrite
import tqdm


if __name__ == '__main__':
    javabridge.start_vm(class_path=bioformats.JARS)
    ### programme starts here ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=Path, default='/mnt/rescomp/users/achatrian/PTEN H&E/V5496 - 1A- PTEN H&E.czi')
    parser.add_argument('--tile_size', type=int, default=256)
    args = parser.parse_args()
    md_seq = bioformats.get_omexml_metadata(path=str(args.file_path))
    metadata = xmltodict.parse(md_seq)
    level0 = metadata['OME']['Image'][0]['Pixels']  # access pixel info for level 0
    width, height = int(level0['@SizeX']), int(level0['@SizeY'])
    tiff_store = np.zeros((height, width, 3), dtype=np.uint8)  # pixel encoding needs to be <32bit for tiff
    xs, ys = list(range(0, width, args.tile_size)), list(range(0, height, args.tile_size))
    with bioformats.ImageReader(str(args.file_path)) as reader:
        for x, y in tqdm.tqdm(product(xs, ys), total=len(xs)*len(ys)):
            if x == xs[-1]:
                x = width - x
            if y == ys[-1]:
                y = height - y
            tile = reader.read(XYWH=(x, y, args.tile_size, args.tile_size))
            tile = tile * 255  # RGB is from 0 to 1 in .czi files
            tiff_store[y:y+args.tile_size, x:x+args.tile_size] = tile.astype(np.uint8)
    save_path = args.file_path.with_suffix('.tiff')
    imwrite(str(args.file_path.with_suffix('.tiff')), tiff_store, bigtiff=True)
    ### programme ends here
    javabridge.kill_vm()
    print("Saved image at {}".format(str(save_path)))


