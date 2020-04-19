from pathlib import Path
from argparse import ArgumentParser
from random import sample
from tempfile import SpooledTemporaryFile
import numpy as np
import cv2
from data.contours import get_contour_image, read_annotations
from data.images.wsi_reader import WSIReader


r"""Compute stain normalization matrix for each slide w.r.to a target slide"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--target_slide', type=str)
    parser.add_argument('--outer_layer', type=str)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--matrix_size', type=int, default=16384)
    args = parser.parse_args()

    image_paths = []
    image_paths += list(path for path in Path(args.data_dir).glob('*.ndpi'))
    image_paths += list(path for path in Path(args.data_dir).glob('*.svs'))
    image_paths += list(path for path in Path(args.data_dir).glob('*.tiff'))
    image_paths += list(path for path in Path(args.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(args.data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(args.data_dir).glob('*/*.tiff'))

    # compute the stain matrix for the target slide
    contour_struct = read_annotations(args.data_dir, slide_ids=(args.target_slide,),
                                      experiment_name=args.experiment_name)
    num_contours = (args.matrix_size/args.tile_size)**2
    target_contours = sample(contour_struct[args.target_slide], num_contours)
    reader = WSIReader()
    with SpooledTemporaryFile(mode='w') as matrix_file:
        matrix = np.memmap(matrix_file, mode='w')
        for contour in target_contours:
            image = get_contour_image(contour, reader)
            too_narrow = image.shape[1] < args.tile_size
            too_short = image.shape[0] < args.tile_size
            if too_narrow or too_short:
                # pad if needed
                delta_w = args.tile_size - image.shape[1] if too_narrow else 0
                delta_h = args.tile_size - image.shape[0] if too_short else 0
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)
            if image.shape[0] > args.tile_size or image.shape[1] > args.tile_size:
                image = image[:args.tile_size, :args.tile_size]






