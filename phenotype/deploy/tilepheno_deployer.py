from pathlib import Path
from numbers import Integral
import logging
from queue import Empty
from datetime import datetime
import imageio
from base.deploy.base_deployer import BaseDeployer


class TilePhenoDeployer(BaseDeployer):

    def __init__(self, opt):
        """Applies network to tiles and converts mask into paths"""
        super(TilePhenoDeployer, self).__init__(opt)
        self.worker_name = 'Tileseg'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--aida_project_name', default='')
        parser.add_argument('--sync_timeout', default=60, type=int, help="Queue time out when putting and getting before error is thrown")
        return parser

    def name(self):
        return "TileSegDeployer"

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
        while True:
            data = input_queue.get(timeout=opt.sync_timeout)
            if data is None:
                input_queue.task_done()
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            output_queue.put((visuals['input_image'], visuals['output_label']))

    @staticmethod
    def gather(deployer, output_queue, sync=()):
        # TODO implement Korsuk's mosaic viz
        pass
