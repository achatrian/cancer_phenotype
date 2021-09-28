from pathlib import Path
from numbers import Integral
from queue import Empty
from base.deploy.base_deployer import BaseDeployer
from annotation.mask_converter import MaskConverter
from annotation.annotation_builder import AnnotationBuilder
from base.utils import utils
# if __debug__:
#     from matplotlib import pyplot as plt
#     import numpy as np
#     import cv2
#     #from base.datasets.wsi_reader import make_wsi_reader, add_reader_args, get_reader_options


################## OBSOLETE ######################


class AnnConvertDeployer(BaseDeployer):

    def __init__(self, opt):
        """Turn masks into aida annotations for one slide"""
        super(AnnConvertDeployer, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--aida_project_name', default='')
        parser.add_argument('--min_contour_area', default=10000, help="Minimum area for object in mask to be converted into contour")
        parser.add_argument('--merge_segments', action='store_true', help="Whether to perform segment merge in gatherer process")
        parser.add_argument('--closeness_threshold', default=5.0, help="Max distance for two points to be considered on boudnary betweem two contours")
        parser.add_argument('--dissimilarity_threshold', default=4.0, help="Max average distance of close points for bounds to be merged")
        parser.add_argument('--max_iter', default=3, help="Max number of iterations in annotation merger")
        parser.add_argument('--sync_timeout', default=60, type=int, help="Queue time out when putting and getting before error is thrown")
        parser.add_argument('--rescale_factor', default=0.0, type=float, help="Resize factor for contours")
        if __debug__:
            parser.add_argument('--slide_file', default='', type=Path)
        parser.set_defaults(model='None')  # no neural net is needed
        return parser

    def name(self):
        return "AnnConvertDeployer"

    @staticmethod
    def run_worker(process_id, opt, model, input_queue, output_queue=None):
        #if __debug__:
            #slide = make_wsi_reader(opt, opt.slide_file)
        # cannot send to cuda outside process in pytorch < 0.4.1 -- patch (torch.multiprocessing issue)
        print("Process {} runs on gpus {}".format(process_id, opt.gpu_ids))
        converter = MaskConverter(
            min_contour_area=opt.min_contour_area)  # set up converter to go from mask to annotation path
        # end patch
        i, num_images = 0, 0
        while True:
            data = input_queue.get()
            if data is None:
                input_queue.task_done()
                break
            j = 0
            for map_, slide_id, x_offset, y_offset in zip(
                    data['target'], data['slide_id'], data['x_offset'], data['y_offset']):
                contours, labels, boxes = converter.mask_to_contour(map_, x_offset, y_offset,
                                                                    rescale_factor=None if not opt.rescale_factor else opt.rescale_factor)
                output_queue.put((contours, labels, boxes), timeout=opt.sync_timeout)
            num_images += data['input'].shape[0]
            if i % opt.print_freq == 0:
                print("[{}] has converted {} tiles".format(process_id, num_images))
            input_queue.task_done()
        output_queue.put(process_id)

    @staticmethod
    def gather(deployer, output_queue, sync=()):
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
            if n_contours % 50 == 0:
                print("Gatherer stored {} contours".format(n_contours))
            i += 1
            output_queue.task_done()
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
        print(f"Saved to {str(save_path)}")




