from pathlib import Path
from base.deploy.base_deployer import BaseDeployer
from base.utils.annotation_converter import AnnotationConverter
from base.utils.aida_annotation import AIDAnnotation
from base.utils import utils


class AnnConvertDeployer(BaseDeployer):

    def __init__(self, opt):
        super(AnnConvertDeployer, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--aida_project_name', default='')
        return parser

    def name(self):
        return "AnnConvertDeployer"

    @staticmethod
    def run_worker(process_id, opt, model=None, queue=None):
        # cannot send to cuda outside process in pytorch < 0.4.1 -- patch (torch.multiprocessing issue)
        print("Process {} runs on gpus {}".format(process_id, opt.gpu_ids))
        converter = AnnotationConverter()  # set up converter to go from mask to annotation path
        # end patch
        i, num_images = 0, 0
        annotations = {}
        # FIXME - multiple processes will create different files for same slide -- and overwrite when dumping -- need intraprocess communication
        if opt.ndeploy_workers > 1:
            import warnings
            warnings.warn('Disjoint annotation files if more than one worker is used (must be fixed)')
        while True:
            data = queue.get()
            if data is None:
                queue.task_done()
                break
            for map_, slide_id, offset_x, offset_y in zip(
                    data['target'], data['slide_id'], data['x_offset'], data['y_offset']):
                if slide_id not in annotations:
                    annotations[slide_id] = AIDAnnotation(slide_id, opt.aida_project_name, ['epithelium', 'lumen', 'background'])
                contours, labels = converter.mask_to_contour(map_, offset_x, offset_y)
                for contour, label in zip(contours, labels):
                    annotations[slide_id].add_item(label, 'path')
                    contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
                    annotations[slide_id].add_segments_to_last_item(contour)
            num_images += data['input'].shape[0]
            if i % opt.print_freq == 0:
                print("[{}] has stored {} annotations".format(process_id, num_images))
            queue.task_done()
        # dump all the annotation objects to json
        save_path = Path(opt.data_dir)/'data'/'annotations'
        utils.mkdirs(str(save_path))
        for annotation in annotations.values():
            annotation.dump_to_json(save_dir=save_path)
        print(f"[{process_id}] Dumped to {str(save_path)}")


