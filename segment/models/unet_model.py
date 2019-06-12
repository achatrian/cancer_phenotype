import warnings
import numpy as np
import torch
from base.models.base_model import BaseModel
import base.models.networks as network_utils
from .networks import UNet


class UNetModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.module_names = ['']
        self.net = UNet(opt.depth, opt.num_class, opt.input_channels, opt.num_filters, opt.patch_size, opt.max_multiple,
                        multiples=[int(m) for m in opt.filter_multiples.split(',')] if opt.filter_multiples else None)
        self.loss_names = ['ce', 'reg'] if self.opt.regularizer_coeff else ['ce']
        self.ce = torch.nn.CrossEntropyLoss(opt.loss_weight, reduction='mean')
        self.reg = network_utils.RegularizationLoss()
        self.metric_names = ['acc', 'dice'] + \
                            ['acc{}'.format(c) for c in range(self.opt.num_class)] + \
                            ['dice{}'.format(c) for c in range(self.opt.num_class)]
        self.visual_names = ["input", "output", "target"]
        self.visual_types = ["image", "map", "map"]
        if self.is_train:
            self.optimizers = [torch.optim.Adam([
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
                 'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
                 'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
            ])]
        self.input, self.target, self.output, self.image_paths = (None,)*4


    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--depth", type=int, default=6, help="number of down-samplings in encoder network")
        parser.add_argument("--max_multiple", type=int, default=32, help="max multiple of the given base number of filter in networks")
        parser.add_argument("--filter_multiples", type=str, default='', help="OR give multiples yourself as comma separated numbers e.g. '1,2,4' (need depth + 1 numbers)")
        parser.add_argument("--regularizer_coeff", type=float, default=0)
        return parser

    def name(self):
        return "UNetModel"

    def set_input(self, data):
        super().set_input(data)
        self.visual_paths = {'input': data['input_path'],
                             'output': [''] * len(data['input_path'])}  # and 3 must be returned by dataset
        if not self.opt.is_apply:
            self.visual_paths['target'] = data['target_path']  # 4 is optional, only for when available

    def forward(self):
        # use set_input() to assign input to model
        if self.input is None:
            raise ValueError("Input not set for {}".format(self.name()))
        self.output = self.net(self.input)
        if self.target is not None:
            self.u_('ce', self.ce(self.output, self.target))
            if self.opt.regularizer_coeff:
                self.u_('reg', self.reg(self.net) * self.opt.regularizer_coeff)
        elif self.opt.is_train:
            warnings.warn("Empty target assigned to model", UserWarning)

    def backward(self):
        self.optimizers[0].zero_grad()
        if self.opt.regularizer_coeff:
            (self.loss_ce + self.loss_reg).backward()
        else:
            self.loss_ce.backward()
        self.optimizers[0].step()

    def optimize_parameters(self):
        self.set_requires_grad(self.net, requires_grad=True)
        self.forward()
        self.backward()

        for scheduler in self.schedulers:
            # step for schedulers that update after each iteration
            try:
                scheduler.batch_step()
            except AttributeError:
                pass

    def evaluate_parameters(self):
        r""" Computes metrics, works for single channel too
            Input is numpy in torch tensor form: NxCxHxW
            Deals both with float gt targets and class labels gt targets
        """
        output = self.output.detach()
        if output.shape[1] > 1:
            output = torch.nn.functional.softmax(output, dim=1)
        else:
            output = torch.nn.functional.sigmoid(output, dim=1)
        output = output.cpu().numpy()
        target = self.target.detach().cpu().numpy()
        class_acc, class_dice = [], []
        for c in range(output.shape[1]):
            pred = output[:, c, ...].flatten()
            gt = target[:, c, ...].flatten() if target.shape[1] == output.shape[1] else (target == c).astype(np.float).flatten()
            class_acc.append(round(float(np.mean(np.array(pred.round() == gt))), 2))
            class_dice.append(round(float(network_utils.dice_coeff(pred, gt)), 2))
        acc = float(np.mean(class_acc))
        dice = float(np.mean(class_dice))
        self.u_('acc', acc)
        self.u_('dice', dice)
        for c in range(len(class_acc)):
            self.u_('metric_acc{}'.format(c), class_acc[c])
            self.u_('metric_dice{}'.format(c), class_dice[c])
