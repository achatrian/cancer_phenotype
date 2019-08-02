import torch
from base.models.base_model import BaseModel
from .networks import IoUEstimator
from base.utils import utils

# TODO must build regression network that takes UNet bottom layer as input. Features from bottom layers will be saved when applying UNet


class IoUEstimatorModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.module_names = ['']  # Must remember this !!!
        self.net = IoUEstimator() # TODO must finish to implement small estimator network
        self.loss_names = ['bce']
        self.mse = torch.nn.MSELoss()
        self.metric_names = ['mse']
        self.visual_names = ['input']
        self.visual_types = ['image']
        if self.is_train:
            self.optimizers = [torch.optim.Adam([
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
                 'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
                 'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
            ])]
        self.input, self.target, self.output, self.loss_mse = (None,) * 4
        self.metric_acc, self.metric_dice, self.metric_auc = (None,) * 3
        self.meters = dict(acc=utils.AverageMeter(), dice=utils.AverageMeter())

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(patch_size=299)
        return parser

    def name(self):
        return "InceptionModel"

    def forward(self):
        if self.input is None:
            raise ValueError("Input not set for {}".format(self.name()))
        self.output = self.net(self.input)
        self.loss_mse = self.mse(self.output, self.target)

    def backward(self):
        self.optimizers[0].zero_grad()
        self.loss_mse.backward()
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


