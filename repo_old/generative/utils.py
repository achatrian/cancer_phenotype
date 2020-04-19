import os
import os.path as ntpath
import sys
import argparse
from PIL import Image

import numpy as np
from scipy.misc import imresize
from scipy import stats
import torch
from torchvision import transforms
import cv2

def on_cluster():
    import socket, re
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)

if on_cluster():
    sys.path.append(os.path.expanduser('~') + '/cancer_phenotype')
else:
    sys.path.append(os.path.expanduser('~') + '/Documents/Repositories/cancer_phenotype')

from generate.util import util
from generate import data
from generate import models


def save_images(visuals, image_paths, image_dir, aspect_ratio=1.0):
    short_paths = [ntpath.basename(image_path) for image_path in image_paths]
    names = [os.path.splitext(short_path)[0] for short_path in short_paths]

    for label, im_data in visuals.items():
        for idx in range(len(names)):
            im = util.tensor2im(im_data[idx:idx+1])
            image_name = '%s_%s.png' % (names[idx], label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)


def denormalize(value):
    return value * 0.5 + 0.5


def img2segmap(gts, return_tensors=False, size=128):

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


def segmap2img(segmaps, return_tensors=False, size=256):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    totensor = transforms.ToTensor()
    if isinstance(segmaps, torch.Tensor):
        segmaps = segmaps.cpu().numpy()
    if segmaps.ndim == 3:
        segmaps = segmaps.transpose(1, 2, 0)
        segmaps = [segmaps]  # to make function work for single image too
    else:
        segmaps = segmaps.transpose(0, 2, 3, 1)
    segmap_store = []
    for segmap in segmaps:
        segmap_1d = np.uint8(segmap[:, :, 3] >= 0.5) * 250
        segmap_1d[np.logical_and(segmap[:, :, 3] < 0.5, segmap[:, :, 2] >= 0.5)] = 200
        segmap_1d[np.logical_and(segmap[:, :, 3] < 0.5, segmap[:, :, 1] >= 0.5)] = 160
        if segmap_1d.shape[-1] != size:
            segmap_1d = imresize(segmap_1d, (size, size), interp='bicubic')
        segmap = segmap_1d[:, :, np.newaxis].repeat(3, axis=2)
        if return_tensors:
            segmap = Image.fromarray(segmap).convert('RGB')
            segmap = totensor(segmap).float()
            segmap = normalize(segmap)
        segmap_store.append(segmap)
    return torch.stack(segmap_store, dim=0) if return_tensors else np.stack(segmap_store, axis=0)


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, default='cycle_gan',
                            help='chooses which model to use. cycle_gan, pix2pix, test')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')

        # TODO trim down unnecessary ones above
        # My options
        parser.add_argument('-vd', '--vaegan_file', type=str, required=True)
        parser.add_argument('-dd', '--data_dir', type=str,
                            default="/gpfs0/well/rittscher/users/achatrian/cancer_phenotype/Dataset")
        parser.add_argument('-sd', '--save_dir', type=str, default='')
        #parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')

        parser.add_argument('--image_size', type=int, default=512)
        parser.add_argument('--max_img_num', type=int, default=10000)
        parser.add_argument('--num_samples', type=int, default=4)
        parser.add_argument('--shuffle', action="store_true")

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.print_options(opt)
        self.opt = opt
        return self.opt

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        #  Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
