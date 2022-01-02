from pathlib import Path
from argparse import ArgumentParser
import json
import re
from datetime import datetime
from random import seed
from openslide import OpenSlideError
from tqdm import tqdm
from data.images.wsi_reader import make_wsi_reader

import faulthandler
faulthandler.enable()  # philips sdk causes can segmentation faults on standard errors

seed(1)

r"""Create tissue masks from WSI thumbnails using simple operators on images"""


# try script on patches to see if it works
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--thumbnail_mpp', type=float, default=8.0)
    parser.add_argument('--target_dir', type=Path, default=None)
    parser.add_argument('--no_recursive_search', action='store_true')
    args = parser.parse_args()
    if args.target_dir is not None:
        thumbnails_dir = args.target_dir
    else:
        thumbnails_dir = args.data_dir/'data'/'thumbnails' if args.target_dir is None else args.target_dir
    thumbnails_dir.mkdir(exist_ok=True, parents=True)
    image_paths = []
    image_paths += list(path for path in Path(args.data_dir).glob('*.ndpi'))
    image_paths += list(path for path in Path(args.data_dir).glob('*.svs'))
    image_paths += list(path for path in Path(args.data_dir).glob('*.tiff'))
    image_paths += list(path for path in Path(args.data_dir).glob('*.isyntax'))
    if not args.no_recursive_search:
        image_paths += list(path for path in Path(args.data_dir).glob('*/*.ndpi'))
        image_paths += list(path for path in Path(args.data_dir).glob('*/*.svs'))
        image_paths += list(path for path in Path(args.data_dir).glob('*/*.tiff'))
        image_paths += list(path for path in Path(args.data_dir).glob('*/*.isyntax'))
    failed_image_reads = []
    try:
        with open(thumbnails_dir / 'thumbnails_info.json', 'r') as thumbnails_info_file:
            thumbnails_info = json.load(thumbnails_info_file)
    except (FileNotFoundError, json.JSONDecodeError):
        thumbnails_info = {}
    for image_path in tqdm(image_paths):
        if image_path.suffix not in {'.tiff', '.isyntax'}:
            continue
        thumbnail_path = thumbnails_dir/f'thumbnail_{image_path.with_suffix("").name}.png'
        slide_id = re.sub(r'\.(ndpi|svs|tiff|isyntax)', '', image_path.name)
        print(slide_id)
        if slide_id == '1215_5C_HE':
            continue
        try:
            wsi = make_wsi_reader(image_path, args)
        except (OpenSlideError, IOError) as err:
            failed_image_reads.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Error occurred when opening slide '{slide_id}'"
            })
            continue
        except RuntimeError as err:
            message = f"Possible PixelEngine error for '{slide_id}' - check for other runtime errors"
            failed_image_reads.append({
                'file': str(image_path),
                'error': str(err),
                'message': message
            })
            print(message)
            continue
        read_level, read_mpp = wsi.find_best_level(args.thumbnail_mpp)
        # assert np.unique(mask).size > 1, "Mask must contain multiple values"
        thumbnails_info[slide_id] = dict(level_downsamples=wsi.level_downsamples,
                                         level_dimensions=wsi.level_dimensions,
                                         mpp=wsi.mpp,
                                         target_mpp=args.thumbnail_mpp,
                                         read_mpp=read_mpp,
                                         read_level=read_level)
    with open(thumbnails_dir / 'thumbnails_info.json', 'w') as thumbnails_info_file:
        json.dump(thumbnails_info, thumbnails_info_file)
    with open(thumbnails_dir/f'failed_image_reads_foreground_extraction_{str(datetime.now())[:10]}.json', 'w') as failures_file:
        json.dump(failed_image_reads, failures_file)
    print("Done!")

