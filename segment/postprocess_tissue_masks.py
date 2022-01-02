from pathlib import Path
from argparse import ArgumentParser
from imageio import imread, imwrite
import json
from numpy import uint8
from tqdm import tqdm
from skimage.transform import rescale
from annotation.mask_converter import MaskConverter
from base.utils.debug import show_image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--masks_dir', type=Path, default="/well/rittscher/projects/ProMPT_data/masks/tissue")
    parser.add_argument('--plot_images', action='store_true')
    parser.add_argument('--small_hole_size', type=int, default=60000)
    # parser.add_argument('--rescale_factor', type=float, default=0.3)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    masks_paths = list(args.masks_dir.glob('*.png')) + list(args.masks_dir.glob('*.jpg'))
    save_dir = args.masks_dir.parent/(args.masks_dir.name + '_postprocess')
    save_dir.mkdir(exist_ok=True)
    with open(args.mask_dir.parent/'thumbnails_info.json', 'r') as thumbnails_info_file:
        masks_info = json.load(thumbnails_info_file)
    for mask_path in tqdm(masks_paths):
        slide_id = mask_path.name[:-4]
        save_path = save_dir/mask_path.name
        if save_path.exists() and not args.overwrite:
            continue
        mask = imread(mask_path)
        if args.plot_images:
            show_image(mask, 'original')
        mask = MaskConverter.remove_ambiguity(mask, initial_opening_size=3, dist_threshold=0.1,
                                              small_hole_size=args.small_hole_size, small_object_size=50, final_closing_size=0,
                                              final_dilation_size=0)
        # if args.rescale_factor != 1.0:
        #     mask = rescale(mask, args.rescale_factor, preserve_range=True)
        # masks_info[slide_id]['postprocessing'] = {
        #     'small_hole_size': args.small_hole_size,
        #     'rescale_factor': args.rescale_factor
        # }
        if args.plot_images:
            show_image(mask, 'postprocessed')
        imwrite(save_path, (mask*255).astype(uint8))
    print("Done!")




