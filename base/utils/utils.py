from __future__ import print_function
import sys
from pathlib import Path
import errno
import time
import socket
import re
from collections import OrderedDict, defaultdict
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from argparse import ArgumentTypeError
import torch
import numpy as np
from PIL import Image


# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123
'''
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://msdn.microsoft.com/en-us/library/windows/desktop/ms681382%28v=vs.85%29.aspx
    Official listing of all such codes.
'''


def is_pathname_valid(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?


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
        raise ArgumentTypeError('Boolean value expected.')


def get_time_stamp():
    date_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    return date_string


def split_options_string(opt_string, splitter=','):
    opts = opt_string.split(f'{splitter}')
    return [int(opt) if str_is_int(opt) else opt for opt in opts]


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
        os.mkdir(str(dir_name))
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
    __slots__ = ['val', 'avg', 'sum', 'count']

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(gt):
    if gt.shape[0] > 1:
        if gt.min() < 0 or gt.max() > 1:  #if logit
            if np.any(gt > 500): gt /= gt.max()
            gt = np.exp(gt) / np.repeat(np.exp(gt).sum(axis=0)[np.newaxis,...], gt.shape[0], axis=0)
        gt = np.round(gt)

        if gt.shape[0] == 3:
            r = np.floor((gt[0, ...] == 1) * 255)
            g = np.floor((gt[1, ...] == 1) * 255)
            b = np.floor((gt[2, ...] == 1) * 255)
        elif gt.shape[0] == 2:
            r = np.floor((gt[1, ...] == 1) * 255)
            g = np.floor((gt[0, ...] == 1) * 255)
            b = np.zeros(gt[0, ...].shape)
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


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, segmap=False, num_classes=3, imtype=np.uint8, visual=True):
    r"""
    Converts image to tensor for visualisation purposes
    :param input_image:
    :param segmap:
    :param num_classes
    :param imtype:
    :param visual: whether output is destined for visualization or processing (3c vs 1c)
    :return: image
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()  # taking the first image only NO MORE
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    if segmap:
        image_numpy = segmap2img(image_numpy, num_classes=num_classes)
    else:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        # for segmentation maps with four classes
    if image_numpy.ndim == 2 and visual:
        image_numpy = image_numpy[:, :, np.newaxis].repeat(3, axis=2)
    return image_numpy.astype(imtype)


def segmap2img(segmap, num_classes=None):
    """
    color coding segmap into an image
    """
    if len(segmap.shape) > 2:
        # multichannel segmap, one channel per class
        segmap = segmap.transpose(1, 2, 0)
        image = np.argmax(segmap, axis=2)
        if segmap.shape[2] == 4:
            image[image == 1] = 160
            image[image == 2] = 200
            image[image == 3] = 250
        elif segmap.shape[2] == 3:
            image[image == 1] = 200
            image[image == 2] = 250
        elif segmap.shape[2] == 2:
            image[image == 1] = 250
        else:
            raise ValueError("Conversion of map to image not supported for shape {}".format(segmap.shape))
    elif num_classes:
        num_labels = len(np.unique(segmap))
        if num_labels > num_classes:
            raise ValueError(f"More labels than classes in segmap ({num_labels} > {num_classes}")
        if num_classes == 2:
            segmap *= 2
        elif num_classes == 3:
            segmap[segmap == 1] = 200
            segmap[segmap == 2] = 250
        elif num_classes == 4:
            segmap[segmap == 1] = 160
            segmap[segmap == 2] = 200
            segmap[segmap == 3] = 250
        else:
            raise NotImplementedError(f"Can't handle {num_classes} classes")
        image = segmap
    else:
        raise ValueError('For single channel segmap, num_classes must be > 0')
    return image


# TODO fix this if needed - not updated in a while
def img2segmap(gts, return_tensors=False, size=128):
    """
    !!! Currently only works for pix2pix version !!!
    :param gts:
    :param return_tensors:
    :param size:
    :return:
    """

    def denormalize(value):
        # undoes normalization that is applied in pix2pix aligned dataset
        return value * 0.5 + 0.5

    gts = denormalize(gts)

    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().numpy()
    if gts.ndim == 3:
        gts = gts.transpose(1, 2, 0)
        gts = [gts]  # to make function work for single image too
    else:
        gts = gts.transpose(0, 2, 3, 1)

    gt_store, label_store = [], []
    for gt in gts:
        if gt.shape[0:2] != (size,)*2:
            gt = cv2.resize(gt, dsize=(size,)*2)
        label = stats.mode(gt[np.logical_and(gt > 0, gt != 250)], axis=None)[
            0]  # take most common class over gland excluding lumen
        if label.size > 0:
            label = int(label)
            if np.isclose(label*255, 160):
                label = 0
            elif np.isclose(label*255, 200):
                label = 1
        else:
            label = 0.5

        gt[np.isclose(gt*255, 160, atol=45)] = 40/255  # to help get better map with irregularities introduced by augmentation

        # Normalize as wh en training network:

        gt = gt[:, :, 0]
        gt = np.stack((np.uint8(np.logical_and(gt >= 0, gt < 35/255)),
                       np.uint8(np.logical_and(gt >= 35/255, gt < 45/255)),
                       np.uint8(np.logical_and(gt >= 194/255, gt < 210/255)),
                       np.uint8(np.logical_and(gt >= 210/255, gt <= 255/255))), axis=2)

        if return_tensors:
            gt = torch.from_numpy(gt.transpose(2, 0, 1)).float()
            label = torch.tensor(label).long()
        gt_store.append(gt)
        label_store.append(label)

    gts = torch.stack(gt_store, dim=0) if return_tensors else np.stack(gt_store, axis=0)
    labels = torch.stack(label_store, dim=0) if return_tensors else np.stack(label_store, axis=0)
    return gts, labels


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
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


def summary(model, input_size, device, batch_size=-1):
    """
    Prints out a detailed summary of the pytorch model.
    From: https://github.com/sksq96/pytorch-summary
    :param model:
    :param input_size:
    :param batch_size:
    :param device:
    :return:
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        model.cuda()
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    # return summary


# Saliency maps utils
"""
Created on Thu Oct 21 11:09:09 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import copy
import numpy as np
from PIL import Image
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models


def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def save_gradient_images(gradient, file_path):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
        save_dir (str): path to saving location
    """
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    # Save image
    file_path = Path(file_path).with_suffix('.json')
    mkdir(file_path)
    save_image(gradient, file_path)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name+'_Cam_Heatmap.png')
    print(np.max(heatmap))
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    print()
    print(np.max(heatmap_on_image))
    path_to_file = os.path.join('../results', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # Save grayscale heatmap
    print()
    print(np.max(activation_map))
    path_to_file = os.path.join('../results', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def save_image(im, path):
    """
        Saves a numpy matrix of shape D(1 or 3) x W x H as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image

    TODO: Streamline image saving, it is ugly.
    """
    if isinstance(im, np.ndarray):
        if len(im.shape) == 2:
            im = np.expand_dims(im, axis=0)
            print('A')
            print(im.shape)
        if im.shape[0] == 1:
            # Converting an image with depth = 1 to depth = 3, repeating the same values
            # For some reason PIL complains when I want to save channel image as jpg without
            # additional format in the .save()
            print('B')
            im = np.repeat(im, 3, axis=0)
            print(im.shape)
            # Convert to values to range 1-255 and W,H, D
        # A bandaid fix to an issue with gradcam
        if im.shape[0] == 3 and np.max(im) == 1:
            im = im.transpose(1, 2, 0) * 255
        elif im.shape[0] == 3 and np.max(im) > 1:
            im = im.transpose(1, 2, 0)
        else:
            raise ValueError(f"Invalid array dimensions {im.shape} for image data")
        im = Image.fromarray(im.astype(np.uint8))
    im.save(path)


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        pil_im.thumbnail((512, 512))
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

