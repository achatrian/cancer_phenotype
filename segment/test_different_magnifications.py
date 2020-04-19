from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
import socket
from random import sample
import warnings
import re
from tqdm import tqdm
import numpy as np
import torch
from imageio import imwrite
from options.apply_options import ApplyOptions
from models import create_model
from datasets import create_dataset, create_dataloader
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter
from base.utils.utils import segmap2img


"""
Script used to test network on images at different resolutions and
observe performance changes
"""


if __name__ == '__main__':
    opt = ApplyOptions().parse(False)
    parser = ArgumentParser()
    parser.add_argument('--mpps', type=float, nargs='+', default=[1.0])
    parser.add_argument('--num_tiles', type=int, default=20)
    args = parser.parse_args()
    print(f"Starting at {str(datetime.now())}")
    print(f"Running on host: '{socket.gethostname()}'")
    if not hasattr(opt, 'mpp'):
        raise AttributeError("Dataset does not scale images using mpp parameters -- cannot change magnification")
    # hard-code some parameters for test
    opt.no_visdom = True
    dataset = create_dataset(opt)
    dataset.setup()  # NB swapped in position .make_subset() and .setup()
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    converter = MaskConverter()
    magnification_test_dir = Path(opt.data_dir, 'data', 'magnification_test', opt.experiment_name)
    magnification_test_dir.mkdir(exist_ok=True, parents=True)
    tile_indices = sample(list(range(len(dataset))), min(args.num_tiles, len(dataset)))
    for tile_index in tqdm(tile_indices):
        for mpp in args.mpps:
            dataset.opt.mpp = mpp
            model.set_input(dataset[tile_index])
            model.test()
            output_logits = model.get_current_visuals()['input_image']
            output = torch.nn.functional.softmax(output_logits, dim=0)
    # TODO finish
