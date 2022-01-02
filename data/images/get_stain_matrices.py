from pathlib import Path
from argparse import ArgumentParser
from random import shuffle, sample
from tempfile import SpooledTemporaryFile
from itertools import product, cycle
import multiprocessing as mp
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from imageio import imwrite, imread
from data.contours import get_contour_image, read_annotations
from data.images.wsi_reader import make_wsi_reader
from staintools import MacenkoStainExtractor
from base.utils import debug


r"""Compute stain normalization matrix for each slide w.r.to a target slide"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--outer_layer', type=str, default='epithelium')
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--matrix_size', type=int, default=2**13)
    parser.add_argument('--annotations_dir', type=Path, default=None)
    parser.add_argument('--save_dir', type=Path, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shuffle_annotations', action='store_true')
    parser.add_argument('--suffix', action='append', default=['ndpi', 'svs', 'tiff', 'isyntax'])
    parser.add_argument('--recursive_search', action='store_true')
    parser.add_argument('--slides_list_path', type=str, default=None, help="Path to file containing subset of slides ids to process")
    # wsi reader args
    parser.add_argument('--mpp', default=0.40, type=float, help="MPP value to read images from slide")  # CHANGED DEFAULT FROM 0.5 to 0.4
    args = parser.parse_args()
    if args.annotations_dir is None:
        args.annotations_dir = args.data_dir
        full_path = False
    else:
        full_path = True
    if args.save_dir is None:
        args.save_dir = Path(args.data_dir, 'data   ', f'{args.outer_layer}:stain_references', args.experiment_name)
    args.save_dir.mkdir(exist_ok=True, parents=True)
    (args.save_dir/'references').mkdir(exist_ok=True, parents=True)
    image_paths = []
    for suffix in args.suffix:
        image_paths += list(path for path in Path(args.data_dir).glob(f'*.{suffix}'))
        if args.recursive_search:
            image_paths += list(path for path in Path(args.data_dir).glob(f'*/*.{suffix}'))
    if args.shuffle_annotations:
        shuffle(image_paths)
    if args.slides_list_path is not None:
        slides_list = pd.read_excel(Path(args.slides_list_path))
        slides_identifiers = set(str(id_) for id_ in slides_list['SlideIdentifier'])
        image_paths = [image_path for image_path in image_paths if image_path.stem in slides_identifiers]
        print(f"Selected {len(image_paths)} slides from {args.slides_list_path}")

    def save_stain_matrix(image_path):
        slide_id = image_path.with_suffix('').name
        try:
            reference = imread(args.save_dir/'references'/f'{slide_id}.png')
            stain_matrix = MacenkoStainExtractor().get_stain_matrix(reference)
            np.save(args.save_dir / f'{slide_id}.npy', stain_matrix)
            return
        except (ValueError, FileNotFoundError, SyntaxError):
            # PIL throws a syntax error for corrupted pngs: PIL/PngImagePlugin.py", line 166, in read raise SyntaxError(f"broken PNG file (chunk {repr(cid)})")
            pass
        contour_struct = read_annotations(args.annotations_dir, slide_ids=(slide_id,),
                                          experiment_name=args.experiment_name, full_path=full_path)

        num_contours = (args.matrix_size//args.tile_size)**2
        try:
            if len(contour_struct[slide_id]) == 0:
                print(f"No annotation for {slide_id}")
                return
            elif any(len(layer_contours) == 0 for layer_contours in contour_struct[slide_id].values()):
                print(f"Annotation for slide {slide_id} is empty")
                return
            contours = contour_struct[slide_id][args.outer_layer]
            sample_contours = sample(contours, min(num_contours, len(contours)))
        except KeyError as err:
            print(err)
            return
        reader = make_wsi_reader(image_path, {'patch_size': args.tile_size, 'mpp': args.mpp})
        with SpooledTemporaryFile() as matrix_file:
            reference = np.memmap(matrix_file, mode='w+', dtype=np.uint8,
                                        shape=(args.matrix_size, args.matrix_size, 3))
            for contour, (i, j) in tqdm(zip(cycle(sample_contours),
                                            product(range(int(num_contours**(1/2))),
                                                    range(int(num_contours**(1/2))))),
                                        desc=f"reading contours for {slide_id} ...", total=num_contours):
                image = get_contour_image(contour, reader, min_size=(args.tile_size,)*2).astype(np.uint8)
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
                reference[i*args.tile_size:(i+1)*args.tile_size, j*args.tile_size:(j+1)*args.tile_size] = image
        imwrite(args.save_dir/'references'/f'{slide_id}.png', reference)
        stain_matrix = MacenkoStainExtractor().get_stain_matrix(reference)
        np.save(args.save_dir/f'{slide_id}.npy', stain_matrix)
        print(f"Stain reference and stain matrix saved for {slide_id}")

    pbar = tqdm(total=len(image_paths), desc="Estimating stain matrix ...")

    def update(*a):
        pbar.update()

    # compute the stain matrix for the target slide
    if args.workers > 0:
        with mp.Pool(args.workers) as pool:
            result = pool.map_async(save_stain_matrix, image_paths, callback=update)
            result.get()
    else:
        for image_path in image_paths:
            save_stain_matrix(image_path)
            pbar.update()





