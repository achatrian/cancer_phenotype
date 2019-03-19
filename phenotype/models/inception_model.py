from contextlib import contextmanager
import torch
from .networks import Inception
from sklearn.metrics import confusion_matrix
import numpy as np
from base.models.base_model import BaseModel
from base.utils import utils


class InceptionModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.module_names = ['']  # Must remember this !!!
        self.net = Inception(opt.num_class)
        self.loss_names = ['bce']
        self.bce = torch.nn.CrossEntropyLoss(opt.loss_weight, reduction='mean')
        self.metric_names = ['acc', 'dice']
        self.visual_names = ["input", 'prediction', 'target']
        self.visual_types = ["image", 'label', 'label']

        if self.is_train:
            self.optimizers = [torch.optim.Adam([
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
                 'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
                 'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
            ])]

        self.input = None
        self.target = None
        self.output = None
        self.prediction = None
        self.loss_bce = None
        self.metric_acc = None
        self.metric_dice = None
        self.meters = dict(acc=utils.AverageMeter(), dice=utils.AverageMeter())

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return "InceptionModel"

    def forward(self):
        if self.input is None:
            raise ValueError("Input not set for {}".format(self.name()))
        self.output = self.net(self.input)
        self.update_measure_value('bce', self.bce(self.output, self.target))

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_bce.backward()
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
        EPS = 0.01
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
