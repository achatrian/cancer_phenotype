import os
from math import ceil
import time
import socket
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from argparse import Namespace #for get_flags

def str_is_int(s):
    r"""
    Check if string is convertable to an integer
    """
    try:
        int(s)
        return True
    except ValueError:
        return False

def str2bool(v):
    r"""
    Use with argparse to convert string to bool
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_time_stamp():
    date_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string

def on_cluster():
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)

def bytes2human(n):
    # http://code.activestate.com/recipes/578019
    # >>> bytes2human(10000)
    # '9.8K'
    # >>> bytes2human(100001221)
    # '95.4M'
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = dict()
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '{:.2f}{}B'.format(value, s)
    return "{}B".format(n)


def check_mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass

def get_flags(filepath):
    with open(filepath, 'r') as argsfile:
        args = eval(argsfile.readline())
    return args


#### torch

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def evaluate_multilabel(predictions, gts):
    r""" Computes metrics, works for single channel too
        Input is numpy in torch tensor form: NxCxHxW
    """
    acc_cls, dice_cls = [],[]
    if predictions.shape[1] > 1:
        predictions = np.exp(predictions) / np.repeat(
                        np.exp(predictions).sum(axis=1)[:,np.newaxis,...], predictions.shape[1], axis=1)
    else:
        predictions = 1/(1 + np.exp(-predictions))
    for c in range(predictions.shape[1]):
        pred = predictions[:,c,...].flatten()
        gt = gts[:,c,...].flatten()
        acc_cls.append(round(float(np.mean(np.array(pred.round() == gt))),2))
        dice_cls.append(round(float(dice_coeff(pred, gt)), 2))
    acc = float(np.mean(acc_cls))
    dice = float(np.mean(dice_cls))
    return acc, acc_cls, dice, dice_cls

def dice_coeff(pred, target):
    r"""This definition generalize to real valued pred and target vector.
    Exact - for numpy arrays
    """
    smooth = 0.0001
    iflat = pred.flatten().round()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    return ((2. * intersection + smooth) / (tflat.sum() + iflat.sum() + smooth))

def dice_loss(output, target, weights):
    """
    output : NxCxHxW Variable
    target :  NxCxHxW FloatTensor
    weights : C FloatTensor
    """
    eps = 0.0001

    intersection = output * target
    numerator = 2 * intersection.sum(0).sum(1).sum(1) + eps
    denominator = output + target
    denominator = denominator.sum(0).sum(1).sum(1) + eps
    loss_per_channel = weights * (1 - (numerator / denominator))
    return loss_per_channel.sum() / output.size(1)

class MultiLabelSoftDiceLoss(nn.Module):
    def __init__(self, weights=None, num_class=3):
        super(MultiLabelSoftDiceLoss, self).__init__()
        if num_class>1:
            self.sm = nn.Softmax2d()
        else:
            self.sm = nn.Sigmoid()
        self.weights = nn.Parameter(torch.from_numpy(np.array(weights) or np.array([1 for i in range(num_class)])).type(torch.FloatTensor),
                        requires_grad=False)

    def forward(self, outputs, targets):
        return dice_loss(self.sm(outputs), targets, self.weights)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(gt):
    if gt.shape[0] > 1:
        if gt.min() < 0 or gt.max() > 1: #if logit
            if np.any(gt > 500): gt /= gt.max()
            gt = np.exp(gt) / np.repeat(np.exp(gt).sum(axis=0)[np.newaxis,...], gt.shape[0], axis=0)
        gt = np.round(gt)

        if gt.shape[0] == 3:
            r = np.floor((gt[0,...] == 1) * 255)
            g = np.floor((gt[1,...] == 1) * 255)
            b = np.floor((gt[2,...] == 1) * 255)
        elif gt.shape[0] == 2:
            r = np.floor((gt[1,...] == 1) * 255)
            g = np.floor((gt[0,...] == 1) * 255)
            b = np.zeros(gt[0,...].shape)
        elif gt.shape[0] == 1:
            r,g,b = gt*255, gt*255, gt*255
        else:
            raise NotImplementedError
    else:
        if gt.min() < 0 or gt.max() > 1:
            gt = 1/(1+np.exp(-gt))
        gt = np.round(gt)[0,...]
        r,g,b = gt*255, gt*255, gt*255
    gt_colorimg = np.stack([r,g,b], axis=2).astype(np.uint8)
    return gt_colorimg

def get_instances(gt):
    return cv2.connectedComponents(gt)[1] #0th tuple element is number of instances


#Weight map for segmentation
class WeightMap():

    def __init__(self, weight, border=0.0, small_objs=0.0, epithelial=0.0):
        """
        Create object to weight the pixel-wise loss function on the basis of three criteria:
        - Border weight based on distance from two closest gland instances (outside of instances)
        - Weight based on gland size (the smaller the higher)
        - Higher weight if epithelium / lumen ratio is high

        """
        self.weight = 0.0
        self.border = 0.0
        self.small_objs = 0.0
        self.epithelial = 0.0
        assert(0.0 <= self.border <= 1.0) #param is fro 0 to 1
        assert(0.0 <= self.small_objs <= 1.0) #param is fro 0 to 1
        assert(0.0 <= self.epithelial <= 1.0) #param is fro 0 to 1

    def build_map(self, gt):
        pass


#### NOT USED ###

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index =-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate_singlelabel(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes)) #confusion matrix
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) #intersection over union?
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


class PolyLR(object):
    def __init__(self, optimizer, curr_iter, max_iter, lr_decay):
        self.max_iter = float(max_iter)
        self.init_lr_groups = []
        for p in optimizer.param_groups:
            self.init_lr_groups.append(p['lr'])
        self.param_groups = optimizer.param_groups
        self.curr_iter = curr_iter
        self.lr_decay = lr_decay

    def step(self):
        for idx, p in enumerate(self.param_groups):
            p['lr'] = self.init_lr_groups[idx] * (1 - self.curr_iter / self.max_iter) ** self.lr_decay

    def forward(self, x):
        x_shape = x.size()  # (b, c, h, w)
        offset = self.offset_filter(x)  # (b, 2*c, h, w)
        offset_w, offset_h = torch.split(offset, self.regular_filter.in_channels, 1)  # (b, c, h, w)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
        if not self.input_shape or self.input_shape != x_shape:
            self.input_shape = x_shape
            grid_w, grid_h = np.meshgrid(np.linspace(-1, 1, x_shape[3]), np.linspace(-1, 1, x_shape[2]))  # (h, w)
            grid_w = torch.Tensor(grid_w)
            grid_h = torch.Tensor(grid_h)
            if self.cuda:
                grid_w = grid_w.cuda()
                grid_h = grid_h.cuda()
            self.grid_w = nn.Parameter(grid_w)
            self.grid_h = nn.Parameter(grid_h)
        offset_w = offset_w + self.grid_w  # (b*c, h, w)
        offset_h = offset_h + self.grid_h  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3])).unsqueeze(1)  # (b*c, 1, h, w)
        x = F.grid_sample(x, torch.stack((offset_h, offset_w), 3))  # (b*c, h, w)
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))  # (b, c, h, w)
        x = self.regular_filter(x)
        return x

def sliced_forward(single_forward):
    def _pad(x, crop_size):
        h, w = x.size()[2:]
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, pad_h, pad_w

    def wrapper(self, x):
        batch_size, _, ori_h, ori_w = x.size()
        if self.training and self.use_aux:
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            aux_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            for s in self.scales:
                new_size = (int(ori_h * s), int(ori_w * s))
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
                scaled_x = Variable(scaled_x).cuda()
                scaled_h, scaled_w = scaled_x.size()[2:]
                long_size = max(scaled_h, scaled_w)
                print(scaled_x.size())

                if long_size > self.crop_size:
                    count = torch.zeros((scaled_h, scaled_w))
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    aux_outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    stride = int(ceil(self.crop_size * self.stride_rate))
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
                    for yy in range(h_step_num):
                        for xx in range(w_step_num):
                            sy, sx = yy * stride, xx * stride
                            ey, ex = sy + self.crop_size, sx + self.crop_size
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)
                            print(x_sub.size())
                            outputs_sub, aux_sub = single_forward(self, x_sub)

                            if sy + self.crop_size > scaled_h:
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]
                                aux_sub = aux_sub[:, :, : -pad_h, :]

                            if sx + self.crop_size > scaled_w:
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]
                                aux_sub = aux_sub[:, :, :, : -pad_w]

                            outputs[:, :, sy: ey, sx: ex] = outputs_sub
                            aux_outputs[:, :, sy: ey, sx: ex] = aux_sub

                            count[sy: ey, sx: ex] += 1
                    count = Variable(count).cuda()
                    outputs = (outputs / count)
                    aux_outputs = (outputs / count)
                else:
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
                    outputs, aux_outputs = single_forward(self, scaled_x)
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
                    aux_outputs = aux_outputs[:, :, : -pad_h, : -pad_w]
                outputs_all_scales += outputs
                aux_all_scales += aux_outputs
            return outputs_all_scales / len(self.scales), aux_all_scales
        else:
            outputs_all_scales = Variable(torch.zeros((batch_size, self.num_classes, ori_h, ori_w))).cuda()
            for s in self.scales:
                new_size = (int(ori_h * s), int(ori_w * s))
                scaled_x = F.upsample(x, size=new_size, mode='bilinear')
                scaled_h, scaled_w = scaled_x.size()[2:]
                long_size = max(scaled_h, scaled_w)

                if long_size > self.crop_size:
                    count = torch.zeros((scaled_h, scaled_w))
                    outputs = Variable(torch.zeros((batch_size, self.num_classes, scaled_h, scaled_w))).cuda()
                    stride = int(ceil(self.crop_size * self.stride_rate))
                    h_step_num = int(ceil((scaled_h - self.crop_size) / stride)) + 1
                    w_step_num = int(ceil((scaled_w - self.crop_size) / stride)) + 1
                    for yy in range(h_step_num):
                        for xx in range(w_step_num):
                            sy, sx = yy * stride, xx * stride
                            ey, ex = sy + self.crop_size, sx + self.crop_size
                            x_sub = scaled_x[:, :, sy: ey, sx: ex]
                            x_sub, pad_h, pad_w = _pad(x_sub, self.crop_size)

                            outputs_sub = single_forward(self, x_sub)

                            if sy + self.crop_size > scaled_h:
                                outputs_sub = outputs_sub[:, :, : -pad_h, :]

                            if sx + self.crop_size > scaled_w:
                                outputs_sub = outputs_sub[:, :, :, : -pad_w]

                            outputs[:, :, sy: ey, sx: ex] = outputs_sub

                            count[sy: ey, sx: ex] += 1
                    count = Variable(count).cuda()
                    outputs = (outputs / count)
                else:
                    scaled_x, pad_h, pad_w = _pad(scaled_x, self.crop_size)
                    outputs = single_forward(self, scaled_x)
                    outputs = outputs[:, :, : -pad_h, : -pad_w]
                outputs_all_scales += outputs
            return outputs_all_scales

    return wrapper
