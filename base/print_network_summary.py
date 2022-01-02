from pathlib import Path
from argparse import ArgumentParser
from collections import namedtuple
from contextlib import redirect_stdout
from json import load
from models import create_model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoints_dir', type=Path, default='/well/rittscher/users/achatrian/experiments/')
    parser.add_argument('--experiment_name', type=str, default='combined_mpp1.0_normal')
    args = parser.parse_args()
    options_path = args.checkpoints_dir/args.experiment_name/'opt.json'
    summary_path = args.checkpoints_dir/args.experiment_name/'networks_summary.txt'
    with open(options_path, 'r') as options_file:
        options = load(options_file)
    options['verbose'] = True
    options['gpu_ids'] = [3]
    opt = namedtuple('Options', options.keys())(**options)
    with open(summary_path, 'w') as summary_file:
        with redirect_stdout(summary_file):
            model = create_model(opt)
            model.setup()
            model.print_networks(True)
    print("Done!")







