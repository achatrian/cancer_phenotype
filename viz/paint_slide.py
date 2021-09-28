from pathlib import Path
import argparse
from PIL import Image
import math
import numpy as np
import imageio
import pandas as pd
from skimage import transform
from matplotlib import cm
from tqdm import tqdm
from data.images.wsi_reader import make_wsi_reader, add_reader_args, get_reader_options
from base.datasets.base_dataset import CenterCrop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', type=Path, required=True)
    parser.add_argument('--labels_path', type=Path, required=True)
    parser.add_argument('--export_level', type=int, default=3)
    parser.add_argument('--export_dir', type=Path, default='/well/rittscher/users/achatrian/temp')
    parser.add_argument('--blend_alpha', type=float, default=0.5)
    parser.add_argument('--max_save_side', type=int, default=4096, help="Longest side of saved images will be equal or less than this")
    parser.add_argument('--fit_type', type=str, default='rescale', choices=['rescale', 'crop'])
    args = parser.parse_args()
    assert args.labels_path.suffix == '.h5'
    assert args.slide_path.suffix in ('.svs', '.ndpi')
    labels = pd.read_hdf(args.labels_path)
    slide_id = args.slide_path.name[:-5]
    slide_labels = labels.loc[slide_id]
    slide_labels.index = [tuple(int(d) for d in s.split('_')) for s in slide_labels.index]
    cmap_name = 'tab20'
    reader = make_wsi_reader(file_name=args.slide_path)
    downsample = reader.level_downsamples[args.export_level]
    read_start_corner = (min(slide_labels.index, key=lambda bounding_rect: bounding_rect[0])[0],
                         min(slide_labels.index, key=lambda bounding_rect: bounding_rect[1])[1])
    end_horz_rect = max(slide_labels.index, key=lambda bounding_rect: bounding_rect[0] + bounding_rect[2])
    end_vert_rect = max(slide_labels.index, key=lambda bounding_rect: bounding_rect[1] + bounding_rect[3])
    read_end_corner = (end_horz_rect[0] + end_horz_rect[2], end_vert_rect[1] + end_vert_rect[3])  # coord + side len
    read_dimensions = (int((read_end_corner[0] - read_start_corner[0])/downsample),
                       int((read_end_corner[1] - read_start_corner[1])/downsample))
    print(f"Reading ({read_dimensions[1]}, {read_dimensions[0]}) region ...")
    cluster_map = np.array(reader.read_region(read_start_corner, args.export_level, read_dimensions))
    if cluster_map.shape[2] == 4:
        cluster_map = cluster_map[..., :3]
    print("Painting cluster membership over thumbnail ...")
    for (x, y, w, h), label in tqdm(slide_labels.iteritems(), total=len(slide_labels)):
        xs, ys = x - read_start_corner[0], y - read_start_corner[1]  # shift coords to match start of cluster_map
        xd, yd, wd, hd = tuple(math.floor(d/downsample) for d in (xs, ys, w, h))  # rescale bounding rect
        image = cluster_map[yd:yd+hd, xd:xd+wd]
        label_color = np.array(cm.tab20(int(label)))[:3]  # get RGB value of color
        color_mask = (np.ones_like(image) * label_color * 255).astype(np.uint8)
        assert color_mask.ndim == 3 and len(np.unique(color_mask[..., 0])) <= 2
        blend = np.array(Image.blend(
            Image.fromarray(image),
            Image.fromarray(color_mask),
            alpha=args.blend_alpha))  # blend tissue images with color mask
        cluster_map[yd:yd+hd, xd:xd+wd] = blend
    if cluster_map.shape[0] > args.max_save_side or cluster_map.shape[1] > args.max_save_side:
        if args.fit_type == 'rescale':
            rescale_factor = args.max_save_side/max(cluster_map.shape[0], cluster_map.shape[1])
            cluster_map = transform.rescale(cluster_map, rescale_factor, preserve_range=True)
        elif args.fit_type == 'crop':
            cluster_map = CenterCrop(args.max_save_side)(cluster_map)
    imageio.imwrite(args.export_dir/f'{slide_id[:10]}_cluster_map.png', cluster_map.astype(np.uint8))
    print(f"Final map dimensions: {cluster_map.shape} for slide {slide_id}. Done!")
