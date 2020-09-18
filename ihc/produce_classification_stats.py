from pathlib import Path
from argparse import ArgumentParser
import json


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--load_epoch', type=str, default='latest')
    args = parser.parse_args()
    results_dir = args.data_dir/f'{args.experiment_name}_{args.load_epoch}'
    for slide_result_path in results_dir.iterdir():
        if slide_result_path.suffix != '.json':
            continue
        with slide_result_path.open('r') as slide_result_file:
            slide_results = json.load(slide_result_file)


