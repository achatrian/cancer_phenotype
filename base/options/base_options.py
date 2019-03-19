import os
import argparse
import multiprocessing as mp
import torch
from base import models
from base import data
from base import deploy
from base.utils import utils
from options.task_options import get_task_options


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--task', type=str, default='segment', help="Defines structure of problem - to load config of other files")
        parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/rittscher/users/achatrian/ProstateCancer/Dataset")
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--dataset_name', type=str, default="tileseg")
        parser.add_argument('--patch_size', type=int, default=1024, help='crop images to this size')
        parser.add_argument('--fine_size', type=int, default=512, help='then scale to this size --DEPRECATED--')  # FIXME - remove deprecated option
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--model', type=str, default="UNet", help="The network model that will be used")
        parser.add_argument('--eval', action='store_true', help='use eval mode during validation / test time.')
        parser.add_argument('--num_class', type=int, default=3, help='Number of classes to classify the data into')
        parser.add_argument('-nf', '--num_filters', type=int, default=15, help='mcd number of filters for unet conv layers')
        parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
        parser.add_argument('--learning_rate_patience', default=50, type=int)
        parser.add_argument('--weight_decay', default=5e-4, type=float)
        parser.add_argument('--reg_weight', default=5e-4, type=float, help="weight given to regularization loss")
        parser.add_argument('--losstype', default='ce', choices=['dice', 'ce'])
        parser.add_argument('--loss_weight', type=str, default=None)
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids (comma separated numbers - e.g. 1,2,3')
        parser.add_argument('--set_visible_devices', type=utils.str2bool, default='y', help="whether to choose visible devices inside script")
        parser.add_argument('--workers', default=4, type=int, help='the number of workers used to load the data')
        parser.add_argument('--experiment_name', default="experiment_name", type=str)
        parser.add_argument('--checkpoints_dir', default='', type=str, help='checkpoint folder')
        parser.add_argument('--load_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default=0, help='which iteration to load? if load_iter > 0, whether load models by iteration')
        parser.add_argument('-ad', '--augment_dir', type=str, default='')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--fork_processes', action='store_true', help="Set method to create dataloader child processes to fork instead of spawn (could take up more memory)")
        parser.add_argument('--augment_level', type=int, default=0, help='level of augmentation applied to input when training (my_opt)')
        #parser.add_argument('--generated_only', action="store_true") # replace by making dataset

        self.parser = parser
        self.is_train = None
        self.is_apply = None
        self.opt = None

    def gather_options(self):
        # get the basic options
        opt, _ = self.parser.parse_known_args()

        # load task module and task-specific options
        task_name = opt.task
        task_options = get_task_options(task_name)
        parser = task_options.add_actions(self.parser)
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        if model_name and model_name != 'none':
            model_option_setter = models.get_option_setter(model_name, task_name)
            parser = model_option_setter(parser, self.is_train)
            opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_name
        if dataset_name and dataset_name != 'none':
            dataset_option_setter = data.get_option_setter(dataset_name, task_name)
            parser = dataset_option_setter(parser, self.is_train)

        if self.is_apply:
            # modify deployer-related parser options
            deployer_name = opt.deployer_name
            deployer_option_setter = deploy.get_option_setter(deployer_name, task_name)
            parser = deployer_option_setter(parser, self.is_train)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk - only when training or will overwrite training information
        if self.is_train:
            expr_dir = os.path.join(opt.checkpoints_dir, opt.experiment_name)
            utils.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test
        opt.is_apply = self.is_apply
        # check options:
        self.print_options(opt)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0 and opt.set_visible_devices:
            torch.cuda.set_device(opt.gpu_ids[0])
        if opt.loss_weight:
            opt.loss_weight = [float(w) for w in opt.loss_weight.split(',')]
            if len(opt.loss_weight) != opt.num_class:
                raise ValueError("Given {} weights, when {} classes are expected".format(
                    len(opt.loss_weight), opt.num_class))
            else:
                opt.loss_weight = torch.Tensor(opt.loss_weight)
        # set multiprocessing
        if opt.workers > 0 and not opt.fork_processes:
            mp.set_start_method('spawn', force=True)

        self.opt = opt
        return self.opt
