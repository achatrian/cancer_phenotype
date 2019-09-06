# file where to test things
import argparse
from pathlib import Path
from itertools import product
import json
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
    parser.add_argument('-d', '--data_dir', type=Path, required=True)
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--image_list', type=Path, default=None)
    args = parser.parse_args()
    image_counter = 0
    if args.image_list is not None:
        with args.image_list.open('r') as list_file:
            selected_image_ids = []
            for line in list_file:
                line = line.replace('\n', '')
                selected_image_ids.append(line)
    for file_path in args.data_dir.iterdir():
        if file_path.suffix == '.czi' and (args.overwrite or not file_path.with_suffix('.tiff').is_file()):
            if args.image_list is not None:
                if not any(file_path.name.startswith(image_id) for image_id in selected_image_ids):
                    continue
            print("Converting {} ...".format(file_path.name))
            md_seq = bioformats.get_omexml_metadata(path=str(file_path))
            md_dir = file_path.parent / 'metadata'
            if not md_dir.is_dir():
                md_dir.mkdir()
            metadata = xmltodict.parse(md_seq)
            with (md_dir/file_path.with_suffix('.json').name).open(mode='w') as md_file:
                json.dump(metadata, md_file)
            level0 = metadata['OME']['Image'][0]['Pixels']  # access pixel info for level 0
            width, height = int(level0['@SizeX']), int(level0['@SizeY'])
            tiff_store = np.zeros((height, width, 3), dtype=np.uint8)  # pixel encoding needs to be <32bit for tiff
            xs, ys = list(range(0, width, args.tile_size)), list(range(0, height, args.tile_size))
            with bioformats.ImageReader(str(file_path)) as reader:
                for x, y in tqdm.tqdm(product(xs, ys), total=len(xs)*len(ys)):
                    if x == xs[-1]:
                        x = width - x
                    if y == ys[-1]:
                        y = height - y
                    tile = reader.read(XYWH=(x, y, args.tile_size, args.tile_size))
                    tile = tile * 255  # RGB is from 0 to 1 in .czi files
                    tiff_store[y:y+args.tile_size, x:x+args.tile_size] = tile.astype(np.uint8)
            save_path = file_path.with_suffix('.tiff')
            imwrite(str(file_path.with_suffix('.tiff')), tiff_store, bigtiff=True)
            print("Saved images #{} at {}".format(image_counter, str(save_path)))
            image_counter += 1
    ### programme ends here
    javabridge.kill_vm()


