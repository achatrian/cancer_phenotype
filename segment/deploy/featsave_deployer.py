from pathlib import Path
import logging
from datetime import datetime
import json
import torch
import numpy as np
from base.deploy.base_deployer import BaseDeployer


class FeatSaveDeployer(BaseDeployer):

    def __init__(self, opt):
        """Applies network to tiles and converts mask into paths"""
        super().__init__(opt)
        self.worker_name = 'Tileseg'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--aida_project_name', default='')
        parser.add_argument('--sync_timeout', default=60, type=int, help="Queue time out when putting and getting before error is thrown")
        parser.set_defaults(model='UNetFeatExtract', gatherer=False)
        return parser

    def name(self):
        return "FeatSaveDeployer"

    @staticmethod
    def run_worker(process_id, opt, model, input_queue, output_queue=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(Path(opt.checkpoints_dir)/f'apply_{opt.model}_{datetime.now()}.log')  # logging to file for debugging
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()  # logging to console for general runtime info
        ch.setLevel(logging.DEBUG)  # if this is set to ERROR, then errors are not printed to log, and vice versa
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | ID: %(process)d | %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        # cannot send to cuda outside process in pytorch < 0.4.1 -- patch (torch.multiprocessing issue)
        if not hasattr(model, 'is_setup') or not model.is_setup:
            model.setup()  # hence must do setuphere
            model.is_setup = True
            logger.info("Process {} runs on gpus {}".format(process_id, opt.gpu_ids))
        # end patch
        j, num_images = 0, 0
        save_dir = Path(opt.data_dir)/'data'/'feature_maps'/opt.slide_id
        save_dir.mkdir(parents=True, exist_ok=True)
        feature_map_shape_shown = False
        dice_values = dict()
        while True:
            data = input_queue.get(timeout=opt.sync_timeout)
            if data is None:
                input_queue.task_done()
                break
            model.set_input(data)
            model.test()
            model.evaluate_parameters()
            visuals = model.get_current_visuals()
            center_features = model.center.detach().cpu()  # feature maps extracted from bottom layer of the u-net
            dice_scores = np.array(model.class_dice)
            for i, dice, image, feature_map in zip(range(center_features.shape[0]), dice_scores, visuals['input_image'], center_features):
                offset_x, offset_y = data['x_offset'][i], data['y_offset'][i]
                if not feature_map_shape_shown:
                    print(f"Feature map shape is {center_features.shape}")
                    feature_map_shape_shown = True
                torch.save(feature_map, save_dir/f'{offset_x}_{offset_y}.pt')
                dice_values[f'{offset_x}_{offset_y}'] = dice.tolist()
            num_images += data['input'].shape[0]
            if j % opt.print_freq == 0:
                logger.info("[{}] has extracted features from {} tiles".format(process_id, num_images))
            input_queue.task_done()
        with open(save_dir/f'dice_values_tempchunk{process_id}.json', 'w') as dice_values_file:
            json.dump(dice_values, dice_values_file)
        print(f"Subprocess {process_id} has saved {num_images} dice scores and images -- terminating")

    def cleanup(self, output=None):
        r"""Cleanup: Merge all dice_values files into one """
        # FIXME this running before queue is completely joined ?
        save_dir = Path(self.opt.data_dir)/'data'/'feature_maps'/self.opt.slide_id
        chunk_paths = tuple(path for path in save_dir.iterdir() if path.name.startswith('dice_values_tempchunk'))
        dice_values_chunks = []
        for chunk_path in chunk_paths:
            with open(chunk_path, 'r') as chunk_file:
                dice_values_chunks.append(json.load(chunk_file))
            chunk_path.unlink()  # used to remove file or symbolic link
        dice_values = dict()
        for chunk in dice_values_chunks:
            dice_values.update(chunk)
        with open(save_dir/'dice_values.json', 'w') as dice_values_file:
            json.dump(dice_values, dice_values_file)  # save only one file containing all the dice values for the tiles








