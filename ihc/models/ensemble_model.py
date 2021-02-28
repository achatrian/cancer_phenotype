from pathlib import Path
from copy import deepcopy
import numpy as np
import torch.multiprocessing as mp
import torch
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix
from base.models.base_model import BaseModel
from base.models import find_model_using_name, get_option_setter


def apply_net(net, input, bce, target):
    output = net(input)
    loss = bce(output, target)
    return [output.detach(), loss.detach()]


class EnsembleModel(BaseModel):
    r"""
    Class to run inference using several models as an ensemble classifier
    NB works only with 1
    """

    def __init__(self, opt):
        super().__init__(opt)
        if self.opt.is_train:
            raise ValueError("EnsembleModel can only be used in testing or applying mode")
        assert len(self.opt.experiment_name) > 1, "Minimum 2 experiments must be loaded into the ensemble"
        print(f"Ensemble of {len(self.opt.experiments)} networks")
        print("Experiments: " + ",".join(self.opt.experiments))
        self.module_names = self.opt.experiments
        module = find_model_using_name(self.opt.ensemble_module, self.opt.task)  # find models of given networks
        self.models = []
        # assert len(self.opt.gpu_ids) == len(self.opt.experiments), "One gpu must be listed per experiment"
        self.gpus = self.opt.gpu_ids.copy()
        self.gpus = self.gpus * (1 + max(len(self.opt.experiments) - len(self.gpus), 0))  # repeat till each network has at least 1 gpu
        for i, experiment in enumerate(self.opt.experiments):
            model_instance = module(deepcopy(self.opt))
            self.models.append(model_instance)
            net = getattr(model_instance, 'net' + model_instance.module_names[0])
            setattr(self, f'net{experiment}', net)
        self.opt.gpu_ids = []
        self.submodule_name = self.models[0].module_names[0]
        self.bce = torch.nn.CrossEntropyLoss(opt.loss_weight, reduction='mean')
        model = self.models[0]
        self.metric_names = model.metric_names
        self.visual_names = model.visual_names
        self.visual_types = model.visual_types
        self.input, self.target, self.output, self.variance = None, None, None, None
        self.loss_bce, self.loss_variance, self.features = None, None, None
        self.model_outputs, self.model_losses = [], []
        self.pool = mp.Pool(len(self.opt.experiments))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # overwrite the experiment name option with a list like option
        parser.add_argument('--ensemble_module', type=str, default='inception')
        parser.add_argument('--experiments', type=str, action='append', help="Experiments to load")
        return parser

    def name(self):
        return "EnsembleModel"

    # overwrite standard load_networks to enable ensemble loading
    def load_networks(self, epoch):
        for i, experiment in enumerate(self.opt.experiments):
            self.model_tag = f'{epoch}_net'
            load_filename = f'{self.model_tag}_{self.submodule_name}.pth'
            load_path = Path(self.opt.checkpoints_dir, experiment, load_filename)
            net = getattr(self, 'net' + experiment)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print(f'loading the model from {load_path}')
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=torch.device('cpu'))
            # state_dict = dict((key, value.cuda(device=self.device)) for key,value in state_dict.items())
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            # # patch InstanceNorm checkpoints prior to 0.4
            # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            # if not self.opt.gpu_ids:
            #    state_dict = {key[6:]: value for key, value in
            #                    state_dict.items()}  # remove data_parallel's "module."
            net.load_state_dict(state_dict)
            # push to respective gpu:
            net.to(torch.device(self.gpus[i]))

    def forward(self):
        if self.input is None:
            raise ValueError("Input not set for {}".format(self.name()))
        self.model_outputs = []
        self.model_losses = []
        self.model_features = []
        for i, name in enumerate(self.module_names):
            net = getattr(self, 'net' + name)
            output = net(self.input.to(device=self.gpus[i]))
            loss = self.bce(output, self.target.to(device=self.gpus[i]))
            self.model_outputs.append(output.to(device=self.gpus[0]))
            self.model_losses.append(loss.to(device=self.gpus[0]))
            if hasattr(net, 'features'):
                self.model_features.append(net.features.to(device=self.gpus[0]))
            elif hasattr(net.module, 'features'):  # data parallel
                self.model_features.append(net.module.features.to(device=self.gpus[0]))
        # outputs_losses = self.pool.starmap(apply_net,
        #                                    [[getattr(self, 'net' + name), self.input, self.bce, self.target]
        #                                     for name in self.opt.experiments])
        # for output, loss in outputs_losses:
        #     self.model_outputs.append(output.to(device=self.opt.gpu_ids[0]))
        #     self.model_losses.append(loss.to(device=self.opt.gpu_ids[0]))
        self.output = torch.mean(torch.stack(self.model_outputs), 0)
        self.variance = torch.var(torch.stack(self.model_outputs), 0)
        self.loss_bce = torch.mean(torch.stack(self.model_losses), 0)
        self.loss_variance = torch.var(torch.stack(self.model_losses), 0)
        if self.model_features:
            # Fuchs 2020 uses sum of binarized features for deep-sets
            self.features = torch.sum(torch.stack([softmax(features, dim=1) for features in self.model_features]), 0)

    def backward(self):
        raise NotImplementedError("Ensemble model only works in test mode")

    def evaluate_parameters(self):
        EPS = 0.01
        self.update_measure_value('bce', self.loss_bce.detach())  # store loss average
        target = self.target.detach().cpu().numpy()
        output = torch.nn.functional.softmax(self.output, dim=1).max(1)[1].detach().cpu().numpy().astype(target.dtype)
        self.prediction = output  # for visualizer
        cm = confusion_matrix(target, output)
        d = cm.diagonal()
        acc = (d.sum() + EPS) / (cm.sum() + EPS)
        tp = d.sum()
        dice = (2 * tp + EPS) / (2 * tp + (cm - np.diag(d)).sum() + EPS)
        self.update_measure_value('acc', acc)
        self.update_measure_value('dice', dice)
