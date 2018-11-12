#!/usr/bin/python

import os
import sys
import random
import argparse
from pathlib import Path
from numbers import Integral
import warnings

import numpy as np
from torchvision.transforms import ToTensor  #NB careful -- this changes the range from 0:255 to 0:1 !!!
import torchvision.utils as vutils
from torch import no_grad, cuda
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from imageio import imwrite
from inception import Inception, DenseNet

def on_cluster():
    import socket
    import re
    hostname = socket.gethostname()
    match1 = re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname)
    match2 = re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname)
    match3 = re.search("login(\w\w)?.cluster", hostname)
    match4 = re.search("gpu(\w\w)?", hostname)
    match5 = re.search("compG(\w\w\w)?", hostname)
    match6 = re.search("rescomp(\w)?", hostname)
    return bool(match1 or match2 or match3 or match4 or match5)

if on_cluster():
    sys.path.append("/gpfs0/users/win/achatrian/cancer_phenotype")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype")

from ck5.ck5_dataset import ERGGlandDataset
from ck5.ck5_utils import eval_classification, DiceLoss
from segment.utils import on_cluster, get_time_stamp, check_mkdir, str2bool, colorize, AverageMeter, get_flags
from segment.clr import CyclicLR

cudnn.benchmark = True

ckpt_path = ''
writer = None

warnings.filterwarnings("ignore")  # CAREFUL WITH THIS !!!!!!!

def test(test_loader, net, criterion):
    global ckpt_path, writer
    ckpt_path, writer = ckpt_path, writer  # update with global value

    net.eval() #!!

    test_loss = AverageMeter()
    test_preds = []
    test_labels = []
    with no_grad():  # don't track variable history for backprop (to avoid out of memory)
        # NB @pytorch variables and tensors will be merged in the future
        with tqdm(total=len(test_loader)) as pbar:
            for vi, data in enumerate(test_loader):  # pass over whole validation dataset
                inputs, labels = data
                N = inputs.size(0)

                inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
                labels = Variable(labels).cuda() if cuda.is_available() else Variable(labels)
                outputs = net(inputs)
                test_labels.append(labels.cpu().numpy())
                test_preds.append(np.argmax(outputs.data.cpu().numpy(), axis=1))

                # calculate loss
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o, labels) for o in outputs))
                else:
                    loss = criterion(outputs, labels)

                test_loss.update(loss.data.item(), N)
                pbar.update(1)

    # Compute metrics
    test_preds = np.concatenate(test_preds, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    acc, acc_cls, dice, dice_cls = eval_classification(test_preds, test_labels)
    return test_loss.avg, acc, acc_cls, dice, dice_cls


def main(FLAGS):
    global ckpt_path, writer
    ckpt_path, writer = Path(ckpt_path), writer

    if cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral):
            cuda.set_device(FLAGS.gpu_ids)
        else:
            cuda.set_device(FLAGS.gpu_ids[0])

    parallel = not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and cuda.is_available()

    # Model:
    if FLAGS.network_id == "inception_v3":
        net = Inception(num_classes=FLAGS.num_class)
    elif FLAGS.network_id == "densenet169":
        net = DenseNet(num_classes=FLAGS.num_class)
    else:
        raise ValueError("\"{}\" is not a correct model id".format(FLAGS.network_id))
    if cuda.is_available():
        net = net.cuda()
    print("Loaded inception model")

    if parallel:
        net = DataParallel(net, device_ids=FLAGS.gpu_ids).cuda()
    net.eval()

    print('testing ' + str(Path(FLAGS.model_file).name))

    dev = None if cuda.is_available() else 'cpu'
    state_dict = torch.load(FLAGS.model_file, map_location=dev)
    if not parallel:
        state_dict = {key[7:] : value for key, value in state_dict.items()}
    net.load_state_dict(state_dict)

    # Define the loss
    criterion1 = CrossEntropyLoss(size_average=True)
    criterion2 = DiceLoss()
    if cuda.is_available():
        criterion1 = criterion1.cuda()
        criterion2.tocuda()
    criterion = lambda preds, targets : criterion1(preds, targets) + criterion2(preds, targets)
    criterion_val = criterion

    if FLAGS.full_glands:
        test_dataset = ERGGlandDataset(str(FLAGS.data_file), "test", tile_size=FLAGS.image_size, augment=False)
    else:
        test_dataset = ERGDataset(str(FLAGS.data_file), "test", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, drop_last=True, shuffle=True,
                             num_workers=FLAGS.workers)  # batch norm needs to drop last

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Running on GPUs: {}".format(FLAGS.gpu_ids))
    print("Memory: Image size: {}, Batch size: {}".format(FLAGS.image_size, FLAGS.batch_size))
    print("Network has {} parameters".format(params))
    print("Using network {}, {} loss".format(FLAGS.network_id, FLAGS.losstype))
    print("Saving results in {}".format(str(ckpt_path)))
    print("Begin testing ...")

    test_loss, acc, acc_cls, dice, dice_cls = test(test_loader, net, criterion_val)
    test_msg = 'Test result - val loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}'.format(
        test_loss, acc, acc_cls, dice, dice_cls)
    print(test_msg)

    with open(str(Path(FLAGS.model_file).parent/"test_result") + str(Path(FLAGS.model_file).name), 'w') as save_file:
        save_file.write(test_msg)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('-df', '--data_file', type=str, default="/gpfs0/users/win/achatrian/cancer_phenotype/ck5/glands_ERGlabel.csv")

    parser.add_argument('--image_size', type=int, default=299)
    parser.add_argument('--downsample', type=float, default=2.0)
    parser.add_argument('--full_glands', type=str2bool, default="y")

    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--network_id', type=str, default="inception_v3")
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--learning_rate_patience', default=50, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--losstype', default='ce', choices=['dice', 'ce'])

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--val_batch_size', default=40, type=int)
    parser.add_argument('--val_imgs_sample_rate', default=0.05, type=float)

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")

    timestamp = get_time_stamp()
    if on_cluster():
        ckpt_path = "/well/win/users/achatrian/cancer_phenotype/logs/" + "ck5_" + timestamp + "/ckpt"
    else:
        ckpt_path = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Logs"
    exp_name = ''
    visualize = ToTensor()

    main(FLAGS)
