from pathlib import Path
from numbers import Integral
from queue import Empty
from base.deploy.base_deployer import BaseDeployer
from base.utils.annotation_converter import AnnotationConverter
from base.utils.annotation_saver import AnnotationSaver
from base.utils import utils


class AnnConvertDeployer(BaseDeployer):

    def __init__(self, opt):
        """Turn masks into aida annotations for one slide"""
        super(AnnConvertDeployer, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--slide_id', type=str, required=True)  # required slide id
        parser.add_argument('--min_contour_area', default=10000, help="Minimum area for object in mask to be converted into contour")
        parser.add_argument('--closeness_threshold', default=5.0, help="Max distance for two points to be considered on boudnary betweem two contours")
        parser.add_argument('--dissimilarity_threshold', default=4.0, help="Max average distance of close points for bounds to be merged")
        parser.add_argument('--max_iter', default=3, help="Max number of iterations in annotation merger")
        return parser

    def name(self):
        return "AnnConvertDeployer"

    @staticmethod
    def run_worker(process_id, opt, model, input_queue, output_queue=None):
        # cannot send to cuda outside process in pytorch < 0.4.1 -- patch (torch.multiprocessing issue)
        print("Process {} runs on gpus {}".format(process_id, opt.gpu_ids))
        converter = AnnotationConverter(min_contour_area=opt.min_contour_area)  # set up converter to go from mask to annotation path
        # end patch
        i, num_images = 0, 0
        while not input_queue.empty():
            data = input_queue.get()
            if data is None:
                input_queue.task_done()
                break
            for map_, slide_id, offset_x, offset_y in zip(
                    data['target'], data['slide_id'], data['x_offset'], data['y_offset']):
                contours, labels, boxes = converter.mask_to_contour(map_, offset_x, offset_y)
                output_queue.put((contours, labels, boxes))
            num_images += data['input'].shape[0]
            if i % opt.print_freq == 0:
                print("[{}] has converted {} tiles".format(process_id, num_images))
            input_queue.task_done()

    @staticmethod
    def gather(deployer, output_queue, sync=()):
        annotation = AnnotationSaver(deployer.opt.slide_id, deployer.opt.aida_project_name,
                                     ['epithelium', 'lumen', 'background'])
        i, n_contours = 0, 0
        while True:
            if i > 0 and isinstance(data, Integral):
                try:
                    data = output_queue.get(timeout=5)
                except Empty:
                    output_queue.task_done()
                    break
                output_queue.task_done()
            else:
                data = output_queue.get(timeout=60 if i == 0 else 20)
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
        output_queue.join()
        deployer.cleanup(annotation)




