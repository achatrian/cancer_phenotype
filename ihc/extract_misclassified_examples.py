from pathlib import Path
from argparse import ArgumentParser
import json
import re
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
    parser = WSIReader.add_reader_args(parser, include_thresholds=True)
    args = parser.parse_args()
    with open(args.checkpoints_dir/args.experiment_name/'opt.json', 'r') as option_file:
        opt = json.load(option_file)
    image_paths = get_image_paths(opt.data_dir)
    classification_results_dir = args.data_dir/'data'/'classifications'/f'{args.experiment_name}_{args.load_epoch}'
    failure_log = []
    for image_path in image_paths:
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        results_path = classification_results_dir/f'{slide_id}.tiff'
        try:
            with open(results_path, 'r') as results_file:
                slide_results = json.load(results_file)
        except FileNotFoundError as err:
            failure_log.append(str(err))
        slide = WSIReader(image_path, )
        for tile_result in slide_results:








