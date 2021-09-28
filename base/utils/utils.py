from __future__ import print_function
from pathlib import Path
from PIL import Image
import time
import socket
import re
from collections import OrderedDict
from torch import nn
import torch
from argparse import ArgumentTypeError
import numpy as np
import cv2
from torchsummary import summary
from matplotlib import cm
# TODO remove unused functions and clean up existing and used ones !





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


def namespace_to_dict(namespace):
    dict_ = {}
    for k, v in vars(namespace).items():
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        k = str(k)
        dict_[k] = v
    return dict_


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
        r, g, b = gt*255, gt*255, gt*255
    gt_colorimg = np.stack([r, g, b], axis=2).astype(np.uint8)
    return gt_colorimg


# Converts a Tensor into an images array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, segmap=False, num_classes=3, imtype=np.uint8, visual=True):
    r"""
    Converts images to tensor for visualisation purposes
    :param input_image:
    :param segmap:
    :param num_classes
    :param imtype:
    :param visual: whether output is destined for visualization or processing (3c vs 1c)
    :return: images
    """
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()  # taking the first images only NO MORE
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
    r"""
    Converts segmentation maps in a one-class-per-channel or one-value-per-class encodings into visual images
    :param segmap: the segmentation map to convert
    :param num_classes: number of classes in map. If the map is single-channeled, num_classes must be passed and be nonzero
    :return:
    """
    if len(segmap.shape) > 2:
        # multichannel segmap, one channel per class
        if segmap.shape[0] < segmap.shape[1] and segmap.shape[0] < segmap.shape[2]:
            segmap = segmap.transpose(1, 2, 0)
        image = np.argmax(segmap, axis=2)  # turning softmax into class numbers
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
            raise ValueError("Conversion of map to images not supported for shape {}".format(segmap.shape))
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
    return image.astype(np.uint8)


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
        gts = [gts]  # to make function work for single images too
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