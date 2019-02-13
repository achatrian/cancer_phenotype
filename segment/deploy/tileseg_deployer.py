from pathlib import Path
import imageio
import torch
from base.deploy.base_deployer import BaseDeployer
from base.utils import utils
from base.utils.annotation_converter import AnnotationConverter
from base.utils.aida_annotation import AIDAnnotation


class TileSegDeployer(BaseDeployer):

    def __init__(self, opt):
        super(TileSegDeployer, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--aida_project_name', default='')
        return parser

    def name(self):
        return "SegmentDeployer"

    @staticmethod
    def run_worker(process_id, opt, model, queue):
        # cannot send to cuda outside process in pytorch < 0.4.1 -- patch (torch.multiprocessing issue)
        model.opt.gpu_ids = opt.gpu_ids
        model.gpu_ids = opt.gpu_ids
        print("Process {} runs on gpus {}".format(process_id, opt.gpu_ids))
        model.device = torch.device('cuda:{}'.format(model.gpu_ids[0])) if model.gpu_ids else torch.device('cpu')
        model.setup()
        converter = AnnotationConverter()  # set up converter to go from mask to annotation path
        aida_project = AIDAnnotation(opt.aida_project_name, ['epithelium', 'lumen', 'background'])
        # end patch
        if opt.eval:
            model.eval()
        i, num_images = 0, 0
        annotations = {}  # stores AIDA annotation objects
        while True:
            data = queue.get()
            if data is None:
                queue.task_done()
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            outputs = visuals['output_map']
            for map_, slide_id, offset_x, offset_y in zip(
                outputs, data['slide_id'], data['x_offset'], data['y_offset']):
                image = utils.tensor2im(map_, segmap=True)
                if slide_id not in annotations:
                    annotations[slide_id] = AIDAnnotation(slide_id, opt.aida_project_name,
                                                          ['epithelium', 'lumen', 'background'])
                contours, labels = converter.mask_to_contour(image, offset_x, offset_y)
                for contour, label in zip(contours, labels):
                    annotations[slide_id].add_item(label, 'path')
                    contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
                    annotations[slide_id].add_segments_to_last_item(contour)
            num_images += data['input'].shape[0]
            if i % opt.print_freq == 0:
                print("[] has saved {} images".format(process_id, num_images))
            queue.task_done()
        # dump all the annotation objects to json
        save_path = Path(opt.data_dir) / 'data' / 'annotations'
        utils.mkdirs(str(save_path))
        for annotation in annotations.values():
            annotation.dump_to_json(save_dir=save_path)
        print(f"[{process_id}] Dumped to {str(save_path)}")



