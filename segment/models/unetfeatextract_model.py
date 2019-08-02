import warnings
import torch
import numpy as np
from .unet_model import UNetModel
from .networks import UNetFeatExtract
import base.models.networks as network_utils
r"""For extracting Center Features and class-wise IoU from UNet"""


class UNetFeatExtractModel(UNetModel):

    def __init__(self, opt):
        super().__init__(opt)
        self.net = UNetFeatExtract(opt.depth, opt.num_class, opt.input_channels, opt.num_filters, opt.patch_size, opt.max_multiple,
                        multiples=[int(m) for m in opt.filter_multiples.split(',')] if opt.filter_multiples else None)
        self.center = None
        self.class_dice = None

    def forward(self):
        # use set_input() to assign input to model
        if self.input is None:
            raise ValueError("Input not set for {}".format(self.name()))
        self.output, self.center = self.net(self.input)  # difference from superclass is here
        if self.target is not None:
            self.update_measure_value('ce', self.ce(self.output, self.target))
            if self.opt.regularizer_coeff:
                self.update_measure_value('reg', self.reg(self.net) * self.opt.regularizer_coeff)
        elif self.opt.is_train:
            warnings.warn("Empty target assigned to model", UserWarning)

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
        class_acc, class_dice = [], []  # sublist i contains acc/dice for each example for channel i
        for c in range(output.shape[1]):
            class_acc.append([]), class_dice.append([])
            for n in range(output.shape[0]):
                pred = output[n:n+1, c, ...].flatten()
                gt = target[n:n+1, c, ...].flatten() if target.shape[1] == output.shape[1] else (target[n:n+1] == c).astype(np.float).flatten()
                class_acc[-1].append(round(float(np.mean(np.array(pred.round() == gt))), 2))
                class_dice[-1].append(round(float(network_utils.dice_coeff(pred, gt)), 2))
        acc = float(np.mean(class_acc))
        dice = float(np.mean(class_dice))
        self.class_dice = np.array(class_dice).T  # difference from superclass is here -- batch x class
        class_acc, class_dice = np.mean(class_acc, axis=1), np.mean(class_dice, axis=1)
        self.update_measure_value('acc', acc)
        self.update_measure_value('dice', dice)
        for c in range(len(class_acc)):
            self.update_measure_value('acc{}'.format(c), class_acc[c])
            self.update_measure_value('dice{}'.format(c), class_dice[c])



