from pathlib import Path
from argparse import ArgumentParser
import json
import re
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
from openslide import OpenSlideError, OpenSlideUnsupportedFormatError
from data.images.wsi_reader import WSIReader


def get_image_paths(data_dir):
    image_paths = list()
    image_paths += list(path for path in Path(data_dir).glob('*.ndpi'))
    image_paths += list(path for path in Path(data_dir).glob('*.svs'))
    image_paths += list(path for path in Path(data_dir).glob('*.tiff'))
    image_paths += list(path for path in Path(data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(data_dir).glob('*/*.tiff'))
    return image_paths


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--checkpoints_dir', type=Path, default='/well/rittscher/users/achatrian/experiments')
    parser.add_argument('--load_epoch', type=str, default='latest')
    parser.add_argument('--ihc_data_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_2020-04-21.csv')
    parser.add_argument('--data_split', type=Path, default='/well/rittscher/projects/IHC_Request/data/cross_validate/3-split2.json')
    parser.add_argument('--examples_class', type=int, default=0, help="Class of which examples are saved")
    parser.add_argument('--set_mpp', type=float, default=0.25)
    parser.add_argument('--max_tiles_per_slide', type=int, default=10)
    parser = WSIReader.add_reader_args(parser, include_thresholds=True)
    args = parser.parse_args()
    with open(args.checkpoints_dir/args.experiment_name/'opt.json', 'r') as option_file:
        opt = json.load(option_file)
    args.mpp = opt['mpp']
    with open(args.data_split, 'r') as data_split_file:
        train_slides = set(json.load(data_split_file)['train_slides'])
    image_paths = get_image_paths(args.data_dir)
    classification_results_dir = args.data_dir/'data'/'classifications'/f'{args.experiment_name}_{args.load_epoch}'
    if not classification_results_dir.exists():
        raise ValueError(f"No classification results available for experiment {args.experiment_name} at epoch {args.load_epoch}")
    save_dir = args.data_dir/'data'/'misclassified_tiles'/f'{args.experiment_name}_{args.load_epoch}'
    slides_data = pd.read_csv(args.ihc_data_file)
    save_dir.mkdir(exist_ok=True, parents=True)
    failure_log = []
    num_saved_tiles = 0
    for image_path in tqdm(image_paths, desc='slide'):
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        if slide_id not in train_slides:
            continue
        results_path = classification_results_dir/f'{slide_id}.json'
        try:
            with open(results_path, 'r') as results_file:
                slide_results = json.load(results_file)
        except FileNotFoundError as err:
            failure_log.append(str(err))
            continue
        slide_save_dir = save_dir/slide_id
        slide_save_dir.mkdir(exist_ok=True)
        try:
            slide = WSIReader(image_path, args, args.set_mpp)
        except (OpenSlideError, OpenSlideUnsupportedFormatError) as err:
            failure_log.append(str(err))
            continue
        slide_foci_data = slides_data[slides_data['Image'] == slide_id]
        case_type = slide_foci_data.iloc[0]['Case type']
        target = int(case_type == 'Real')
        base_level_patch_size = round(opt['patch_size']*slide.level_downsamples[slide.read_level])
        num_saved_tiles_on_slide = 0
        for tile_result in tqdm(slide_results, 'tile'):
            if num_saved_tiles_on_slide == args.max_tiles_per_slide:
                break
            x, y, classification = tile_result['x'], tile_result['y'], tile_result['classification']
            probabilities = [tile_result[f'probability_class{i}'] for i in range(len(tile_result) - 3)]
            example_class_probability = probabilities[args.examples_class]
            if classification != target == args.examples_class:
                try:
                    tile = np.array(slide.read_region((x, y), 0, (base_level_patch_size,) * 2))[..., :3]
                except OpenSlideError:
                    slide = WSIReader(image_path, args, args.set_mpp)
                    continue
                imageio.imwrite(slide_save_dir/f'{x}_{y}_{classification}_{round(example_class_probability, 2)}.png', tile)
                num_saved_tiles += 1
                num_saved_tiles_on_slide += 1
                if num_saved_tiles % 10 == 0:
                    print(f"Saved {num_saved_tiles} tiles")
    with open(save_dir/'misclassified_info.json', 'w') as misclassified_info_file:
        json.dump({
            'ihc_data_file': str(args.ihc_data_file),
            'data_split': str(args.data_split),
            'examples_class': int(args.examples_class),
            'mpp': float(args.mpp),
            'set_mpp': float(args.set_mpp),
            'patch_size': int(args.patch_size)
        }, misclassified_info_file)
    print(f"Done! {num_saved_tiles} tiles were saved")
    print(f"Errors on {len(failure_log)} tiles")





















