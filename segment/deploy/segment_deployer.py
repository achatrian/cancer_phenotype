import os
import imageio
import torch
from base.deploy.base_deployer import BaseDeployer
from base.utils import utils
from base.utils import json_utils


class SegmentDeployer(BaseDeployer):

    def __init__(self, opt):
        super(SegmentDeployer, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--mask_save_format', default='qupath', type=str, help='Choose how to store annotations')
        parser.set_default(task='segment')
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
        # end patch
        if not opt.is_train:
            if opt.eval:
                model.eval()
            i, num_images = 0, 0
            while True:
                data = queue.get()
                if data is None:
                    queue.task_done()
                    break
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                outputs = visuals['output_map']
                paths = model.get_image_paths()[1]
                for map, path in zip(outputs, paths):
                    image = utils.tensor2im(map, segmap=True)
                    save_path = os.path.join(opt.results_dir, os.path.basename(path))
                    if opt.mask_save_format in ('qupath', 'paperjs'):
                        json_utils.save_annotation_to_json(save_path, image, data['location'][0], data['location'][1],
                                                           annotation=opt.mask_save_format)
                    elif opt.mask_save_format == 'image':
                        imageio.imwrite(save_path, image)
                    else:
                        raise ValueError(f"Unsupported annotation format {opt.mask_save_format} for saving")
                num_images += data['input'].shape[0]
                if i % opt.print_freq == 0:
                    print("[] has saved {} images".format(process_id, num_images))
                queue.task_done()
        else:
            raise NotImplementedError("Training")



