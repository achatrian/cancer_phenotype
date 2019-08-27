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
        self.metric_names = ['acc', 'dice'] + (['auc'] if self.opt.is_train else ['pos_prob'])
        self.visual_names = ["input", 'prediction', 'target']
        self.visual_types = ["image", 'label', 'label']
        if self.is_train:
            self.optimizers = [torch.optim.Adam([
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] == 'bias'],
                 'lr': 2 * opt.learning_rate},  # bias parameters change quicker - no weight decay is applied
                {'params': [param for name, param in self.net.named_parameters() if name[-4:] != 'bias'],
                 'lr': opt.learning_rate, 'weight_decay': opt.weight_decay}  # filter parameters have weight decay
            ])]
        self.input, self.target, self.output = None, None, None
        self.prediction = None
        self.loss_bce = None
        self.metric_acc, self.metric_dice, self.metric_auc = None, None, None
        self.meters = dict(acc=utils.AverageMeter(), dice=utils.AverageMeter())
        self.setup_methods.append('add_per_slide_metrics')
        self.target_store, self.output_store = None, None  # store labels to compute AUC

    # modify parser to add command line options,
    # and also change the default values if needed
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
        self.update_measure_value('bce', self.loss_bce.detach())  # store loss average
        target = self.target.detach().cpu().numpy()
        output = torch.nn.functional.softmax(self.output, dim=1).max(1)[1].detach().cpu().numpy().astype(target.dtype)
        self.prediction = output  # for visualizer
        cm = confusion_matrix(target, output)
        d = cm.diagonal()
        acc = (d.sum() + EPS) / (cm.sum() + EPS)
        tp = d.sum()
        dice = (2 * tp + EPS) / (2 * tp + (cm - np.diag(d)).sum() + EPS)
        output_prob = torch.nn.functional.softmax(self.output, dim=1).detach().cpu().numpy()[:, 1]  # for pos class
        self.update_measure_value('acc', acc)
        self.update_measure_value('dice', dice)
        if self.is_train:
            try:
                auc = roc_auc_score(self.target_store, self.output_store)
            except ValueError:
                # if targets only contain one class AUC is undefined - store more labels and outputs before computing AUC
                if self.target_store is not None and self.target_store.any():
                    self.target_store = np.concatenate((self.target_store, target), 0)
                    self.output_store = np.concatenate((self.output_store, output_prob), 0)
                else:
                    self.target_store = target
                    self.output_store = output_prob
                auc = 0.0
            self.update_measure_value('auc', auc)
            for i, path in enumerate(self.visual_paths['input']):
                slide_name = Path(path).parent.name
                if f'acc_{slide_name}' in self.metric_names:  # not tracking all slides
                    self.update_measure_value(f'acc_{slide_name}', float(target[i] == self.prediction[i]), n_samples=1)
        else:
            pos_prob = np.mean(output)
            self.update_measure_value('pos_prob', pos_prob)  # probability of tile belonging to positive class - for dataset-wide AUC computation

    def add_per_slide_metrics(self, dataset, track_every=5):
        """Setup method - add per-slide metrics"""
        slide_names = sorted(set(Path(path).parent.name for path in dataset.paths))
        slide_names = slide_names[::track_every]
        if self.is_train:
            for slide_name in slide_names:
                metric_name = f'acc_{slide_name}'
                self.metric_names.append(metric_name)
                setattr(self, 'metric_' + metric_name, 0.0)  # so that all can be gathered for viz


