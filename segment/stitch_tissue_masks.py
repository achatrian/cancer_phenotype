# from pathlib import Path
# from argparse import ArgumentParser
# import signal
# import json
# from imageio import imread
# import numpy as np
# import cv2
# import torch
# from skimage.transform import resize
#
# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--data_dir', type=Path)
#     parser.add_argument('--overwrite', action='store_true')
#     parser.add_argument('--plot_images', action='store_true')
#     parser.add_argument('--thumbnail_level', type=int, default=-3)
#     parser.add_argument('--target_dir', type=Path, default=None)
#     parser.add_argument('-ds', '--debug_slide', type=str, action='append',
#                         help='only process slides with specified ids. Useful for debugging')
#     parser.add_argument('--no_recursive_search', action='store_true')
#     parser.add_argument('--shuffle_images', action='store_true')
#     parser.add_argument('--timeout', type=int, default=180)
#     parser.add_argument('--exclude_slide', type=str, action='append')
#     args = parser.parse_args()
#     if args.target_dir is not None:
#         thumbnails_dir = args.target_dir
#     else:
#         thumbnails_dir = args.data_dir/'data'/'thumbnails' if args.target_dir is None else args.target_dir
#     with open(thumbnails_dir / 'thumbnails_info.json', 'r') as thumbnails_info_file:
#         thumbnails_info = json.load(thumbnails_info_file)
#     ### TODO ... finish here ###
#     tile_paths = sorted(tile_paths, key=lambda p: p.name)
#     tissue_mask = None
#     for tile_path in tile_paths:
#         slide_id, x, y, patch_size = tile_path.name.split('_')  # only works if slide_id contains no underscores in name
#         x, y, patch_size = int(x), int(y), int(patch_size)
#         mask_dimensions = reversed(thumbnails_info[slide_id]['level_dimensions'][args.thumbnail_level])
#         if tissue_mask is None:
#             tissue_mask = np.zeroes(mask_dimensions)
#         tile = imread(tile_path)

