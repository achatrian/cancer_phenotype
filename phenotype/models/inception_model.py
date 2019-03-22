from pathlib import Path
import torch
from .networks import Inception
from sklearn.metrics import confusion_matrix, roc_auc_score
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
        self.metric_names = ['acc', 'dice', 'auc']
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
        self.setup_methods.append('add_per_slide_metrics')

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
        try:
            auc = roc_auc_score(target, torch.nn.functional.softmax(self.output, dim=1).detach().cpu().numpy()[:, 1])
        except ValueError:
            auc = 0.5  # if all targets in the batch are of same class, roc_auc_score raises an error
        self.update_measure_value('acc', acc)
        self.update_measure_value('dice', dice)
        self.update_measure_value('auc', auc)
        if self.is_train:
            for i, path in enumerate(self.visual_paths['input']):
                slide_name = Path(path).parent.name
                if f'acc_{slide_name}' in self.metric_names:  # not tracking all slides
                    # FIXME -- not taking average as meant to. Instead it oscillates between 0 and 1
                    self.update_measure_value(f'acc_{slide_name}', float(target[i] == self.prediction[i]), n_samples=1)

    def add_per_slide_metrics(self, dataset, track_every=5):
        """Setup method - add per-slide metrics"""
        slide_names = sorted(set(Path(path).parent.name for path in dataset.paths))
        slide_names = slide_names[::track_every]
        if self.is_train:
            for slide_name in slide_names:
                metric_name = f'acc_{slide_name}'
                self.metric_names.append(metric_name)
                setattr(self, 'metric_' + metric_name, 0.0)  # so that all can be gathered for viz


