import argparse
from pathlib import Path
from itertools import product
from tqdm import tqdm
import numpy as np
from data.images.dzi_io.tile_generator import TileGenerator
from annotation.mask_converter import MaskConverter
from annotation.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--patch_size')
    args = parser.parse_args()

    dzi_dir = Path(args.data_dir)/'data'/'dzi'
    dzi_paths = list(path for path in dzi_dir if path.suffix == '.json')
    converter = MaskConverter(
        value_hier=(0, 255),
        label_value_map={
            'nuclei': 255,
            'background': 0
        },
        label_interval_map={
            'nuclei': (200, 255),
            'background': (0, 200)
        },
        label_options={
            'nuclei': {
                'small_object_size': 50,
                'dist_threshold': 0.001,
                'final_closing_size': 5,
                'final_dilation_size': 2
            }
        }
    )
    for dzi_path in dzi_paths:
        dzi = TileGenerator(str(dzi_path))
        mask_size = dzi.slide_to_mask((args.patch_size,) * 2)[0]
        xs, ys = list(range(0, dzi.width, args.patch_size)), list(range(0, dzi.height, args.patch_size))
        annotation = AnnotationBuilder(dzi_path.with_suffix('').name, 'icr', layers=('background', 'nuclei'))
        print("Processing dzi ...")
        for x, y in tqdm(product(xs, ys), total=len(xs) * len(ys)):
            x_mask, y_mask = dzi.slide_to_mask((x, y))
            if dzi.masked_percent(x_mask, y_mask, mask_size, mask_size) > args.tissue_content_threshold:
                patch = dzi.read_region((x, y), 0, (args.patch_size, args.patch_size), border=0)
                nuclei_patch = np.ones(patch.shape[:2])
                nuclei_patch[patch[..., 1] > 0] = 0  # border
                nuclei_patch[patch[..., 2] > 0] = 250  #
                contours, labels, boxes = converter.mask_to_contour(nuclei_patch, x, y)
                for contour, label in zip(contours, labels):
                    annotation.add_item(label, 'path')
                    contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
                    annotation.add_segments_to_last_item(contour)










