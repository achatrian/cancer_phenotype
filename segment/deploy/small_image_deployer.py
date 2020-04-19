from pathlib import Path
from base.deploy.base_deployer import BaseDeployer
from annotation.mask_converter import MaskConverter
from annotation.annotation_builder import AnnotationBuilder


class SmallImageDeployer(BaseDeployer):

    def __init__(self, opt):
        super().__init__(opt)
        self.worker_name = 'contour_extractor'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return "SmallImageDeployer"

    @staticmethod
    def run_worker(process_id, opt, model, input_queue, output_queue=None):
        model.setup()  # cannot send to cuda outside process in pytorch < 0.4.1 -- patch (torch.multiprocessing issue)
        converter = MaskConverter()
        annotations = {}
        while True:
            data = input_queue.get(timeout=opt.sync_timeout)
            if data is None:
                input_queue.task_done()
                break
            model.set_input(data)
            model.test()
            input_paths = data['input_path']
            visuals = model.get_current_visuals()
            for i, map_ in enumerate(visuals['output_map']):
                input_path = input_paths[i]
                if input_path.name not in annotations:
                    annotation = annotations[input_path.name] = AnnotationBuilder(input_path.name, 'segmentation')
                else:
                    annotation = annotations[input_path.name]
                try:
                    offset_x, offset_y = data['x_offset'][i], data['y_offset'][i]
                except KeyError:
                    offset_x, offset_y = 0, 0
                rescale_factor = float(data['read_mpp'][i] / data['base_mpp'][i]) if 'read_mpp' in data else 2.0
                contours, labels, boxes = converter.mask_to_contour(map_, offset_x, offset_y, rescale_factor)
                for contour, label in zip(contours, labels):
                    annotation.add_item(label, 'path')
                    contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
                    annotation.add_segments_to_last_item(contour)
        save_dir = Path(opt.data_dir)/'annotations'/opt.experiment_name
        for annotation in annotations.values():
            annotation.dump_to_json(save_dir)


