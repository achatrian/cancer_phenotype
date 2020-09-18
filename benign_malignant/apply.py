from pathlib import Path
from datetime import datetime
import socket
import warnings
import re
from tqdm import tqdm
import numpy as np
import torch
from imageio import imwrite
from options.apply_options import ApplyOptions
from models import create_model
from datasets import create_dataset, create_dataloader


if __name__ == '__main__':
    n_examples_per_slide = 5
    opt = ApplyOptions().parse()
    print(f"Starting at {str(datetime.now())}")
    print(f"Running on host: '{socket.gethostname()}'")
    # hard-code some parameters for test
    opt.no_visdom = True
    opt.sequential_samples = True
    suffix_check = lambda path: path.name.endswith('.svs') or path.name.endswith('.ndpi') or \
                                path.name.endswith('.png') or path.name.endswith('.jpg')
    image_paths = list(path for path in Path(opt.data_dir).iterdir() if suffix_check(path))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.png'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.jpg'))
    dataset = create_dataset(opt)
    dataset.setup()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    save_dir = Path(opt.data_dir)/'data'/'experiments'/'benign_malignant'
    save_dir.mkdir(exist_ok=True, parents=True)
    results = {}
    for i, data in enumerate(tqdm(dataloader)):
        model.set_input(data)
        model.test()
        outputs = model.output.detach.cpu().numpy()
        targets = model.target.detach.cpu().numpy()
        for output, target in zip(outputs, targets):
            output = float(output)
            target = int(target)

