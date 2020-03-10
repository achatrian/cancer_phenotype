import socket
from pathlib import Path
import json
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from base.options.test_options import TestOptions
from base.datasets import create_dataset, create_dataloader
from base.models import create_model
r"Test script for network, aggregates results over whole validation dataset"


if __name__ == '__main__':
    opt = TestOptions().parse()
    print(f"Running on host: '{socket.gethostname()}'")
    # hard-code some parameters for test
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.overwrite_split = False
    dataset = create_dataset(opt)
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    attention_map_dir = Path(opt.checkpoints_dir/'attention_maps')
    attention_map_dir.mkdir(exist_ok=True, parents=True)
    for data in dataloader:
        model.set_input(data)
        model.test()
        model.evaluate_parameters()
        visuals = model.get_visuals()

