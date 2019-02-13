import torch
from base.models.base_model import BaseModel
from .networks import Inception
from sklearn.metrics import confusion_matrix
import numpy as np


class InceptionModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.net = Inception(opt.num_class)
        self.loss_names = ['bce']
        self.bce = torch.nn.BCELoss(opt.loss_weight, reduction='elementwise_mean')
        self.metric_names = ['acc', 'dice'] #+ \
                            # ['acc{}'.format(c) for c in range(self.opt.num_class)] + \
                            # ['dice{}'.format(c) for c in range(self.opt.num_class)]
        self.visual_names = ["input"]
        self.visual_types = ["image"]

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
        self.loss_bce = None
        self.metric_acc = None
        self.metric_dice = None

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
        self.loss_bce = self.bce(self.output, self.target)

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
        output = self.output.detach().cpu().numpy()
        cm = confusion_matrix(target, output)
        # C0 = C.copy().fill_diagonal(np.zeros(C.shape[0]))
        d = cm.diagonal()

        acc = (d.sum() + EPS) / (cm.sum() + EPS)
        tp = d.sum()
        dice = (2 * tp + EPS) / (2 * tp + (cm - np.diag(d)).sum() + EPS)
        self.metric_acc = acc
        self.metric_dice = dice
        # class_acc = (d + EPS) / (cm.sum(axis=1) + EPS)
        # class_dice = (2 * d + EPS) / (2 * d + (cm - np.diag(d)).sum(axis=0) + (cm - np.diag(d)).sum(axis=0) + EPS)
        # for c in range(self.opt.num_class):
        #     setattr(self, 'metric_acc{}'.format(c), class_acc[c])
        #     setattr(self, 'metric_dice{}'.format(c), class_dice[c])