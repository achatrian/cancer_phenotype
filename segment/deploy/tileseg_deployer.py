from pathlib import Path
from numbers import Integral
import logging
from queue import Empty
from datetime import datetime
import imageio
from base.deploy.base_deployer import BaseDeployer
from base.utils.mask_converter import MaskConverter
from base.utils.annotation_builder import AnnotationBuilder
from base.utils import utils


class TileSegDeployer(BaseDeployer):

    def __init__(self, opt):
        """Applies network to tiles and converts mask into paths"""
        super(TileSegDeployer, self).__init__(opt)
        self.worker_name = 'Tileseg'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--aida_project_name', default='')
        parser.add_argument('--merge_segments', action='store_true', help="Whether to perform segment merge in gatherer process")
        parser.add_argument('--min_contour_area', default=10000, help="Minimum area of mask objects for them to be converted")
        parser.add_argument('--closeness_threshold', default=5.0, help="Max distance for two points to be considered on boudnary betweem two contours")
        parser.add_argument('--dissimilarity_threshold', default=4.0, help="Max average distance of close points for bounds to be merged")
        parser.add_argument('--max_iter', default=3, help="Max number of iterations in annotation merger")
        parser.add_argument('--save_masks', action='store_true', help="Whether to save the masks as well as the contours")
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
        converter = MaskConverter(min_contour_area=opt.min_contour_area)  # set up converter to go from mask to annotation path
        # end patch
        i, num_images = 0, 0
        if opt.save_masks:
            save_path = Path(opt.data_dir) / 'data' / 'network_outputs' / opt.slide_id
            utils.mkdir(str(save_path))
        while True:
            data = input_queue.get(timeout=opt.sync_timeout)
            if data is None:
                input_queue.task_done()
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            for map_, offset_x, offset_y in zip(
                    visuals['output_map'], data['x_offset'], data['y_offset']):
                contours, labels, boxes = converter.mask_to_contour(map_, offset_x, offset_y)
                output_queue.put((contours, labels, boxes), timeout=opt.sync_timeout)
                if opt.save_masks:
                    mask = utils.tensor2im(map_, segmap=True,
                                           num_classes=converter.num_classes)  # transforms tensors into mask label image
                    imageio.imwrite(save_path / f"{offset_x}_{offset_y}_{opt.experiment_name}.png", mask)
            num_images += data['input'].shape[0]
            if i % opt.print_freq == 0:
                logger.info("[{}] has converted {} tiles".format(process_id, num_images))
            input_queue.task_done()
        output_queue.put(process_id)

    @staticmethod
    def gather(deployer, output_queue, sync=()):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(Path(deployer.opt.checkpoints_dir)/f'apply_{deployer.opt.model}_{datetime.now()}.log')  # logging to file for debugging
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()  # logging to console for general runtime info
        ch.setLevel(logging.DEBUG)  # if this is set to ERROR, then errors are not printed to log, and vice versa
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | ID: %(process)d | %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.info("Start gathering data")
        annotation = AnnotationBuilder(deployer.opt.slide_id, deployer.opt.aida_project_name, ['epithelium', 'lumen', 'background'])
        i, n_contours = 0, 0
        while True:
            if i > 0 and isinstance(data, Integral):
                try:
                    data = output_queue.get(timeout=deployer.opt.sync_timeout)
                except Empty:
                    output_queue.task_done()
                    break
                output_queue.task_done()
            else:
                data = output_queue.get(timeout=deployer.opt.sync_timeout)
            if data == deployer.opt.ndeploy_workers:
                break
            elif isinstance(data, Integral):
                continue
            contours, labels, boxes = data
            for contour, label, box in zip(contours, labels, boxes):
                annotation.add_item(label, 'path', tile_rect=box)
                contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
                annotation.add_segments_to_last_item(contour)
            n_contours += len(contours)
            i += 1
            output_queue.task_done()
            if i % max(deployer.opt.batch_size // 2, 5) == 0:
                logger.info(f"Processed {n_contours} contours")
        output_queue.join()
        deployer.cleanup(annotation)

    def cleanup(self, output):
        annotation = output
        if self.opt.merge_segments:
            annotation.merge_overlapping_segments(closeness_thresh=self.opt.closeness_threshold,
                                                  dissimilarity_thresh=self.opt.dissimilarity_threshold,
                                                  max_iter=self.opt.max_iter,
                                                  parallel=bool(self.opt.workers),
                                                  num_workers=self.opt.workers,
                                                  log_dir=self.opt.checkpoints_dir)
        # dump all the annotation objects to json
        save_path = Path(self.opt.data_dir) / 'data' / 'annotations'
        utils.mkdirs(str(save_path))
        annotation.dump_to_json(save_dir=save_path)
        print(f"Dumped to {str(save_path)}")


