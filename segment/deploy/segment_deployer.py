import os
import imageio
import torch
from base.deploy.base_deployer import BaseDeployer
from base.utils import utils


class SegmentDeployer(BaseDeployer):

    def __init__(self, opt):
        super(SegmentDeployer, self).__init__(opt)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_default(task='segment')
        return parser

    def name(self):
        return "SegmentDeployer"

    @staticmethod
    def run_worker(process_id, opt, model, queue):
        # cannot send to cuda outside process in pytorch < 0.4.1 -- patch
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
                    imageio.imwrite(save_path, image)
                num_images += data['input'].shape[0]
                if i % opt.print_freq == 0:
                    print("[] has saved {} images".format(process_id, num_images))
                queue.task_done()
        else:
            raise NotImplementedError("Training")



