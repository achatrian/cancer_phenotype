from pathlib import Path
import argparse
from PIL import Image
import math
from itertools import product
import warnings
import numpy as np
import imageio
import pandas as pd
from skimage import color, transform
from matplotlib import cm
from tqdm import tqdm
from data.images.wsi_reader import WSIReader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_path', type=Path, required=True)
    parser.add_argument('--labels_path', type=Path, required=True)
    parser.add_argument('--export_level', type=int, default=3)
    parser.add_argument('--tile_size', type=int, default=8192, help="Size of tile at level 0 to color uniformly")
    parser.add_argument('--export_dir', type=Path, default='/well/rittscher/users/achatrian/temp')
    parser.add_argument('--blend_alpha', type=float, default=0.65)
    parser.add_argument('--max_save_side', type=int, default=4096, help="Longest side of saved images will be equal or less than this")
    args = parser.parse_args()
    assert args.labels_path.suffix == '.h5'
    assert args.slide_path.suffix in ('.svs', '.ndpi')
    labels = pd.read_hdf(args.labels_path)
    slide_id = args.slide_path.name[:-5]
    slide_labels = labels.loc[slide_id]
    slide_labels.index = [tuple(int(d) for d in s.split('_')) for s in slide_labels.index]
    cmap_name = 'tab20'
    reader = WSIReader(file_name=args.slide_path)
    downsample = reader.level_downsamples[args.export_level]
    read_start_corner = (min(slide_labels.index, key=lambda bounding_rect: bounding_rect[0])[0],
                         min(slide_labels.index, key=lambda bounding_rect: bounding_rect[1])[1])
    end_horz_rect = max(slide_labels.index, key=lambda bounding_rect: bounding_rect[0] + bounding_rect[2])
    end_vert_rect = max(slide_labels.index, key=lambda bounding_rect: bounding_rect[1] + bounding_rect[3])
    read_end_corner = (end_horz_rect[0] + end_horz_rect[2], end_vert_rect[1] + end_vert_rect[3])  # coord + side len
    read_dimensions = [int((read_end_corner[0] - read_start_corner[0])/downsample),
                       int((read_end_corner[1] - read_start_corner[1])/downsample)]
    read_dimensions[0] += read_dimensions[0] % args.tile_size
    read_dimensions[1] += read_dimensions[1] % args.tile_size
    print(f"Reading ({read_dimensions[1]}, {read_dimensions[0]}) region ...")
    cluster_map = np.array(reader.read_region(read_start_corner, args.export_level, read_dimensions))
    if cluster_map.shape[2] == 4:
        cluster_map = (color.rgba2rgb(cluster_map) * 255).astype(np.uint8)
    print("Painting cluster membership over thumbnail ...")

    def is_contained(child, parent):
        p_at_x = parent[0] <= child[0] <= parent[0] + parent[2]
        p_at_xw = parent[0] <= child[0] + child[2] <= parent[0] + parent[2]
        p_at_y = parent[1] <= child[1] <= parent[1] + parent[3]
        p_at_yh = parent[1] <= child[1] + child[3] <= parent[1] + parent[3]

        c_at_x = child[0] <= parent[0] <= child[0] + child[2]
        c_at_xw = child[0] <= parent[0] + parent[2] <= child[0] + child[2]
        c_at_y = child[1] <= parent[1] <= child[1] + child[3]
        c_at_yh = child[1] <= parent[1] + parent[3] <= child[1] + child[3]
        return (p_at_x and p_at_xw and p_at_y and p_at_yh) or (c_at_x and c_at_xw and c_at_y and c_at_yh)

    index = np.array(tuple(slide_labels.index))
    xs = list(range(read_start_corner[0], read_end_corner[0], args.tile_size))
    ys = list(range(read_start_corner[1], read_end_corner[1], args.tile_size))
    for xt, yt in tqdm(product(xs, ys), total=len(xs)*len(ys)):
        contained_labels = slide_labels[
            np.apply_along_axis(is_contained, 1, index, (xt, yt, args.tile_size, args.tile_size))
        ]
        if len(contained_labels) == 0:
            warnings.warn(f"No labels contained in tile {(xt, yt, args.tile_size, args.tile_size)}, may have to increase tile size (current = {args.tile_size})")
            continue
        label = contained_labels.mode().iloc[0]
        xd, yd, wd, hd = tuple(math.floor(d/downsample) for d in (xt, yt, args.tile_size, args.tile_size))
        image = cluster_map[yd:yd+hd, xd:xd+wd]
        if image.shape[2] == 4:
            image = (color.rgba2rgb(image) * 255).astype(np.uint8)
        label_color = np.array(cm.tab20(int(label)))[:3]  # get RGB value of color
        color_mask = (np.ones_like(image) * label_color * 255).astype(np.uint8)
        assert color_mask.ndim == 3 and len(np.unique(color_mask[..., 0])) <= 2
        blend = np.array(Image.blend(
            Image.fromarray(image),
            Image.fromarray(color_mask),
            alpha=args.blend_alpha))  # blend tissue images with color mask
        if blend.shape[-2] == 4:
            blend = (color.rgba2rgb(blend) * 255).astype(np.uint8)
        cluster_map[yd:yd+hd, xd:xd+wd] = blend
    if cluster_map.shape[0] > args.max_save_side or cluster_map.shape[1] > args.max_save_side:
        rescale_factor = args.max_save_side / max(cluster_map.shape[0], cluster_map.shape[1])
        cluster_map = transform.rescale(cluster_map, rescale_factor, preserve_range=True)
    cluster_map = cluster_map.astype(np.uint8)
    imageio.imwrite(args.export_dir / f'{slide_id[:10]}_cluster_map.jpeg', cluster_map)
    print(f"Final map dimensions: {cluster_map.shape}. Done!")

