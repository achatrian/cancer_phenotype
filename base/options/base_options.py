from pathlib import Path
import argparse
import multiprocessing as mp
import copy
import json
import re
import torch
from base import models
from base import datasets
from base import deploy
from base.utils import utils
from base.options.task_options import get_task_options


def remove_args(parser, args):
    for arg in args:
        for action in parser._actions:
            if vars(action)['option_strings'][0] == arg:
                parser._handle_conflict_resolve(None, [(arg ,action)])
                break


# def make_conflict_proof(option_setter):
#     r"""Option setter can add arguments to parser with existing actions with same argument string (BREAKS for python <=3.6)"""
#     def conflict_proof_option_setter(parser, is_train):
#         parser_ = copy.deepcopy(parser)  # FIXME fails for python < 3.7
#         error = True
#         conflicting_arguments = []
#         while error:
#             try:
#                 remove_args(parser_, conflicting_arguments)
#                 parser_ = option_setter(parser_, is_train)
#                 error = False
#             except argparse.ArgumentError as err:
#                 conflict_arg = re.search(r'argument (--\w*)', str(err)).groups()[0]
#                 conflicting_arguments.append(conflict_arg)
#         remove_args(parser, conflicting_arguments)
#         return option_setter(parser, is_train)
#     return conflict_proof_option_setter


def make_conflict_proof(option_setter):
    r"""Option setter can add arguments to parser with existing actions with same argument string (compatible with python <=3.6)"""
    def conflict_proof_option_setter(parser, is_train):
        option_strings = set(vars(parser)['_option_string_actions'].keys())
        temp_parser = option_setter(argparse.ArgumentParser(), is_train)
        setter_option_strings = set(vars(temp_parser)['_option_string_actions'].keys())
        conflicting_arguments = list(set.intersection(option_strings, setter_option_strings))
        remove_args(parser, conflicting_arguments)
        return option_setter(parser, is_train)
    return conflict_proof_option_setter


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
        parser.add_argument('--do_not_spawn', action='store_false', dest='spawn_processes', help="Set method to create dataloader child processes to fork instead of spawn (could take up more memory)")
        parser.add_argument('--augment_level', type=int, default=0, help='level of augmentation applied to input when training (my_opt)')
        parser.add_argument('--sequential_samples', action='store_true', help="always iterates over data in the same order")
        self.parser = parser
        self.is_train = None
        self.is_apply = None
        self.opt = None

    def gather_options(self, unknown_arg_error=True):
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
            model_option_setter = make_conflict_proof(model_option_setter)
            parser = model_option_setter(parser, self.is_train)
            opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_name
        if dataset_name and dataset_name != 'none':
            dataset_option_setter = datasets.get_option_setter(dataset_name, task_name)
            dataset_option_setter = make_conflict_proof(dataset_option_setter)
            parser = dataset_option_setter(parser, self.is_train)

        if hasattr(opt, 'deployer_name'):
            # modify deployer-related parser options
            deployer_option_setter = deploy.get_option_setter(opt.deployer_name, task_name)
            deployer_option_setter = make_conflict_proof(deployer_option_setter)
            parser = deployer_option_setter(parser, self.is_train)

        self.parser = parser
        return parser.parse_args() if unknown_arg_error else parser.parse_known_args()[0]

    def print_options(self, opt, return_only=False):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        if not return_only:
            print(message)

        # save to the disk - only when training or will overwrite training information
        if self.is_train and not opt.continue_train:
            expr_dir = Path(opt.checkpoints_dir)/opt.experiment_name
            expr_dir.mkdir(exist_ok=True, parents=True)
            file_name = Path(expr_dir)/'opt.txt'
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        return message

    def parse(self, unknown_arg_error=True):
        opt = self.gather_options(unknown_arg_error)
        opt.is_train = self.is_train   # train or test
        opt.is_apply = self.is_apply
        # check options:
        self.print_options(opt)
        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id_ = int(str_id)
            if id_ >= 0:
                opt.gpu_ids.append(id_)
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
        if opt.spawn_processes:
            mp.set_start_method('spawn', force=True)

        self.opt = opt

        if self.is_train:
            # save opt namespace to json
            expr_dir = Path(opt.checkpoints_dir) / opt.experiment_name
            expr_dir.mkdir(exist_ok=True, parents=True)
            options_path = Path(expr_dir) / 'opt.json'
            opt_to_save = vars(copy.deepcopy(self.opt))
            for opt_name in opt_to_save:  # cannot json serialize paths
                if isinstance(opt_to_save[opt_name], Path):
                    opt_to_save[opt_name] = str(opt_to_save[opt_name])
                elif isinstance(opt_to_save[opt_name], torch.Tensor):
                    opt_to_save[opt_name] = opt_to_save[opt_name].detach().cpu().numpy().tolist()
            with open(options_path, 'w') as options_file:
                json.dump(opt_to_save, options_file)
        return self.opt
