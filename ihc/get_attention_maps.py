import socket
from pathlib import Path
import json
import torch
from tqdm import tqdm
import numpy as np
from imageio import imwrite
from base.options.attribution_options import AttributionOptions
from base.datasets import create_dataset, create_dataloader
from base.models import create_model
r"Test script for network, aggregates results over whole validation dataset"


if __name__ == '__main__':
    opt = AttributionOptions().parse()
    print(f"Running on host: '{socket.gethostname()}'")
    # hard-code some parameters for test
    opt.sequential_samples = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.overwrite_split = False
    # IN ORDER TO GET ALL THE MAPS, THE BATCH SIZE MUST BE == 1,
    # AS ONLY THE ATTENTION MAP FOR THE FIRST IMAGE GETS STORED
    opt.batch_size = 1
    dataset = create_dataset(opt)
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    attention_map_dir = Path(opt.checkpoints_dir/opt.experiment_name/'attention_maps')
    attention_map_dir.mkdir(exist_ok=True, parents=True)
    for data in tqdm(dataloader):
        model.set_input(data)
        model.test()
        model.evaluate_parameters()
        visuals = model.get_visuals()
        input_path = Path(data['input_path'][0])
        attention0 = visuals['attention0']
        attention0_path = attention_map_dir/f'{input_path.name}_att0.jpg'
        imwrite(attention0_path, attention0)
        attention1 = visuals['attention1']
        attention1_path = attention_map_dir/f'{input_path.name}_att1.jpg'
        imwrite(attention1_path, attention1)
        attention2 = visuals['attention2']
        attention2_path = attention_map_dir/f'{input_path.name}_att2.jpg'
        imwrite(attention2_path, attention2)
    print("Done!")





