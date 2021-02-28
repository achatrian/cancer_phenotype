from pathlib import Path
from argparse import ArgumentParser
import json
import re
from datetime import datetime
from random import shuffle
import imageio
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from openslide import OpenSlideError
from tqdm import tqdm
from data.images.wsi_reader import WSIReader
from base.utils.debug import show_image


r"""Create tissue masks from WSI thumbnails using simple operators on images"""


# try script on patches to see if it works
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--plot_images', action='store_true')
    parser.add_argument('--thumbnail_level', type=int, default=-3)
    parser.add_argument('--target_dir', type=Path, default=None)
    parser.add_argument('-ds', '--debug_slide', type=str, action='append',
                        help='only process slides with specified ids. Useful for debugging')
    parser.add_argument('--no_recursive_search', action='store_true')
    parser.add_argument('--shuffle_images', action='store_true')
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
    if not args.no_recursive_search:
        image_paths += list(path for path in Path(args.data_dir).glob('*/*.ndpi'))
        image_paths += list(path for path in Path(args.data_dir).glob('*/*.svs'))
        image_paths += list(path for path in Path(args.data_dir).glob('*/*.tiff'))
    if not args.overwrite:
        id_pattern = re.compile(r'\.(ndpi|svs|tiff)')
        image_paths = [image_path for image_path in image_paths
                       if not (thumbnails_dir/f"thumbnail_{id_pattern.sub('', image_path.name)}.png").exists()]
    if args.shuffle_images:
        shuffle(image_paths)
    failed_image_reads = []
    try:
        with open(thumbnails_dir / 'thumbnails_info.json', 'r') as thumbnails_info_file:
            thumbnails_info = json.load(thumbnails_info_file)
    except FileNotFoundError:
        thumbnails_info = {}
    for image_path in tqdm(image_paths):
        if image_path.suffix != '.tiff':
            continue
        thumbnail_path = thumbnails_dir/f'thumbnail_{image_path.with_suffix("").name}.png'
        if thumbnail_path.exists() and not args.overwrite:
            continue
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        try:
            wsi = WSIReader(image_path)
            thumbnail = np.array(wsi.get_thumbnail(wsi.level_dimensions[args.thumbnail_level]))[..., :3]
        except (OpenSlideError, IOError) as err:
            failed_image_reads.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Error occurred when opening slide '{slide_id}'"
            })
            continue
        except IndexError as err:
            failed_image_reads.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Thumbnail has the wrong format for '{slide_id}'"
            })
            continue
        except ValueError as err:
            failed_image_reads.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Thumbnail level is too large '{slide_id}'"
            })
            continue
        if args.plot_images:
            show_image(thumbnail, 'thumbnail')
        image = thumbnail
        imageio.imwrite(thumbnail_path, image)
        # for old images with red background, whiten the background
        image[(image[..., 0] > 170) & (image[..., 1] < 30) & (image[..., 2] < 30)] = 0
        image[(image[..., 0] <= 50) & (image[..., 1] <= 50) & (image[..., 2] <= 50)] = 0
        image[(image[..., 0] >= 230) & (image[..., 1] >= 230) & (image[..., 2] >= 230)] = 0
        image = cv2.medianBlur(image.astype(np.uint8), 7)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(image)
        gray_image = blurred = s
        if np.all(s == 0):
            mask = s
        else:
            mask = (s > threshold_otsu(s)).astype(np.uint8)
        if args.plot_images:
            show_image(mask, 'threshold')
        midsize_kernel = np.ones((3, 3))
        inv_th, inverse_mask = cv2.threshold(blurred, blurred.mean(), 255, cv2.THRESH_BINARY_INV)
        distance_seed = cv2.distanceTransform(inverse_mask.astype(np.uint8), cv2.DIST_L2, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, midsize_kernel)
        mask = cv2.medianBlur(mask, 5) + 100
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, midsize_kernel)
        mask = remove_small_objects(mask, 10)
        if args.plot_images:
            show_image(mask, 'final')
        imageio.imwrite(thumbnails_dir/f'mask_{image_path.with_suffix("").name}.png', mask)
        thumbnails_info[slide_id] = dict(level_downsamples=wsi.level_downsamples,
                                   **{str(k): v if not isinstance(v, Path) else str(v) for k, v in vars(args).items()})
    with open(thumbnails_dir / 'thumbnails_info.json', 'w') as thumbnails_info_file:
        json.dump(thumbnails_info, thumbnails_info_file)
    with open(thumbnails_dir/f'failed_image_reads_foreground_extraction_{str(datetime.now())[:10]}.json', 'w') as failures_file:
        json.dump(failed_image_reads, failures_file)
    print("Done!")

