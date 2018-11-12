#!/usr/bin/python

import os
import sys
import random
import argparse
from pathlib import Path
from numbers import Integral
import pickle
import warnings

import numpy as np
from torchvision.transforms import ToTensor  # NB careful -- this changes the range from 0:255 to 0:1 !!!
import torchvision.utils as vutils
from torch import no_grad, cuda
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch import save, load, stack
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from inception import Inception, DenseNet
from tqdm import tqdm
from imageio import imwrite


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

from ck5.ck5_dataset import CK5AugmentStudyDataset
from ck5.ck5_utils import eval_classification, DiceLoss
from ck5.eval_ck5_net import test
from segment.utils import on_cluster, get_time_stamp, check_mkdir, str2bool, colorize, AverageMeter, get_flags
from segment.clr import CyclicLR

cudnn.benchmark = True
writer = None

warnings.filterwarnings("ignore")  # CAREFUL WITH THIS !!!!!!!

def train(train_loader, net, criterion, optimizer, epoch, writer, print_freq, scheduler=None):

    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        N = inputs.size(0)
        inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
        labels = Variable(labels).cuda() if cuda.is_available() else Variable(labels)

        optimizer.zero_grad() #needs to be done at every iteration

        # run the model
        outputs = net(inputs)

        # calculate loss
        if isinstance(outputs, tuple):
            loss = sum((criterion(o, labels) for o in outputs))
        else:
            loss = criterion(outputs, labels)

        # backward and optimizer
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.batch_step()

        train_loss.update(loss.data.item(), N)

        #Compute training metrics:
        predictions = np.argmax(outputs[0].data.cpu().numpy(), axis=1)
        targets = labels.data.cpu().numpy()
        acc, acc_cls, dice, dice_cls = eval_classification(predictions, targets)

        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)
        writer.add_scalar('train_acc', acc, curr_iter)
        writer.add_scalar('train_dice', dice, curr_iter)
        if i % print_freq == 0:
            tqdm.write('epoch: {}, batch: {} - train - loss: {:.5f}, acc: {:.2f}, acc_cls: {}. dice: {:.2f}, dice_cls: {}'.format(
            epoch, i, train_loss.avg, acc, acc_cls, dice, dice_cls))

    return {'loss': train_loss.avg, 'acc': acc, 'acc_cls': acc_cls, 'dice': dice, 'dice_cls': dice_cls}

def validate(val_loader, net, criterion, optimizer, epoch, ckpt_path, writer, best_record,
             val_imgs_sample_rate=0.05, val_save_to_img_file=False):

    net.eval() #!!

    val_loss = AverageMeter()
    inputs_smpl, targets_smpl, predictions_smpl = [], [], []
    val_preds = []
    val_labels = []
    with no_grad():  # don't track variable history for backprop (to avoid out of memory)
        # NB @pytorch variables and tensors will be merged in the future
        for vi, data in enumerate(val_loader):  # pass over whole validation dataset
            inputs, labels = data
            N = inputs.size(0)

            inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
            labels = Variable(labels).cuda() if cuda.is_available() else Variable(labels)
            outputs = net(inputs)
            val_labels.append(labels.cpu().numpy())
            val_preds.append(np.argmax(outputs.data.cpu().numpy(), axis=1))

            # calculate loss
            if isinstance(outputs, tuple):
                loss = sum((criterion(o, labels) for o in outputs))
            else:
                loss = criterion(outputs, labels)

            val_loss.update(loss.data.item(), N)

            val_num = max(int(np.floor(val_imgs_sample_rate*N)), 2)
            val_idx = random.sample(list(range(N)), val_num)
            for idx in val_idx:
                inputs_smpl.append(inputs[idx,...].data.cpu().numpy())
                targets_smpl.append(labels[idx].data.cpu().numpy())
                predictions_smpl.append(outputs[idx].data.cpu().numpy())
    targets_smpl = np.stack(targets_smpl, axis=0)
    predictions_smpl = np.stack(predictions_smpl, axis=0)

    # Compute metrics
    val_preds = np.concatenate(val_preds, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    acc, acc_cls, dice, dice_cls = eval_classification(val_preds, val_labels)
    print('epoch: {:d}, val loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}'.format(
        epoch, val_loss.avg, acc, acc_cls, dice, dice_cls))

    if dice > best_record['dice']:
        best_record['val_loss'] = val_loss.avg
        best_record['epoch'] = epoch
        best_record['acc'] = acc
        best_record['acc_cls'] = acc_cls
        best_record['dice'] = dice
        best_record['dice_cls'] = dice_cls
        best_record['lr'] = optimizer.param_groups[1]['lr']
        snapshot_name = 'epoch_.{:d}_loss_{:.5f}_acc_{:.5f}_dice_{:.5f}_lr_{:.10f}'.format(
            epoch, val_loss.avg, acc, dice, optimizer.param_groups[1]['lr'])
        save(net.state_dict(), str(ckpt_path/exp_name/(snapshot_name + '.pth')))
        save(optimizer.state_dict(), str(ckpt_path/exp_name/('opt_' + snapshot_name + '.pth')))

        if val_save_to_img_file:
            to_save_dir = os.path.join(str(ckpt_path), exp_name, str(epoch))
            check_mkdir(to_save_dir)

            val_visual = []
            for idx, (input, label, pred) in enumerate(zip(inputs_smpl, targets_smpl, predictions_smpl)):
                input_rgb = input.transpose(1,2,0)
                if val_save_to_img_file:
                    imwrite(os.path.join(to_save_dir, "{}_input.png".format(idx)), input_rgb)
                val_visual.extend([visualize(input_rgb)])
            writer.add_image(snapshot_name, val_visual)

        print('-----------------------------------------------------------------------------------------------------------')
        print("best record (epoch{:d}):, val loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}".format(
            best_record['epoch'], best_record['val_loss'], best_record['acc'], best_record['acc_cls'],
            best_record['dice'], best_record['dice_cls']))
        print('-----------------------------------------------------------------------------------------------------------')
    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    for c, acc_c in enumerate(acc_cls): writer.add_scalar('acc_c{}'.format(c), acc_c, epoch)
    writer.add_scalar('dice', dice, epoch)
    for c, dice_c in enumerate(dice_cls): writer.add_scalar('dice_c{}'.format(c), dice_c, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    net.train() #reset network state to train (e.g. for batch norm)
    return {'loss': val_loss.avg, 'acc': acc, 'acc_cls': acc_cls, 'dice': dice, 'dice_cls': dice_cls}


def run_aug_exp(FLAGS, aug_level):
    global ckpt_path, writer
    ckpt_path, writer = Path(ckpt_path), writer

    if cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral):
            cuda.set_device(FLAGS.gpu_ids)
        else:
            cuda.set_device(FLAGS.gpu_ids[0])

    if FLAGS.checkpoint_folder:
        ckpt_path = Path(FLAGS.checkpoint_folder)  # scope? does this change inside train and validate functs as well?
        try:
            LOADEDFLAGS = get_flags(str(ckpt_path / ckpt_path.parent.name) + ".txt")
            FLAGS.num_filters = LOADEDFLAGS.num_filters
            FLAGS.num_class = LOADEDFLAGS.num_class
        except FileNotFoundError:
            pass

    parallel = not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and cuda.is_available()

    # Model:
    if FLAGS.network_id == "inception_v3":
        net = Inception(num_classes=3)
    elif FLAGS.network_id == "densenet169":
        net = DenseNet(num_classes=3)
    else:
        raise ValueError("\"{}\" is not a correct model id".format(FLAGS.network_id))
    if cuda.is_available():
        net = net.cuda()
    print("Loaded inception model")

    if parallel:
        net = DataParallel(net, device_ids=FLAGS.gpu_ids).cuda()

    net.train()

    if not FLAGS.snapshot:
        curr_epoch = 1
        best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': [0, 0, 0], 'dice': -.001,
                       'dice_cls': [-.001, -.001, -.001], 'lr': FLAGS.learning_rate}
    else:
        print('training resumes from ' + FLAGS.snapshot)

        dev = None if cuda.is_available() else 'cpu'
        state_dict = load(str(ckpt_path / exp_name / FLAGS.snapshot), map_location=dev)
        if not parallel:
            state_dict = {key[7:]: value for key, value in state_dict.items()}
        net.load_state_dict(state_dict)
        split_snapshot = FLAGS.snapshot.split('_')
        curr_epoch = int(split_snapshot[1][1:]) + 1
        best_record = {'epoch': int(split_snapshot[1][1:]), 'val_loss': float(split_snapshot[3]),
                       'acc': float(split_snapshot[5]),
                       'acc_cls': [float(split_snapshot[5]) for i in range(FLAGS.num_class)],
                       'dice': float(split_snapshot[7]),
                       'dice_cls': [float(split_snapshot[7]) for i in range(FLAGS.num_class)],
                       'lr': float(split_snapshot[9][:-4])}
        val_loss = best_record['val_loss']

    # criterion = BCEWithLogitsLoss(size_average=reduce, reduce=reduce, weight=FLAGS.class_weights)
    criterion1 = CrossEntropyLoss(size_average=True, weight=torch.tensor([0.1, 1, 1]))
    criterion2 = DiceLoss(weight=torch.tensor([0.1, 1, 1]))
    if cuda.is_available():
        criterion1 = criterion1.cuda()
        criterion2 = criterion2.cuda()
    criterion = lambda preds, targets: criterion1(preds, targets) + criterion2(preds, targets)

    # criterion_val = BCEWithLogitsLoss(size_average=True, reduce=reduce, weight=FLAGS.class_weights)
    criterion_val = criterion

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * FLAGS.learning_rate},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': FLAGS.learning_rate, 'weight_decay': FLAGS.weight_decay}
    ])

    if FLAGS.snapshot:
        optimizer.load_state_dict(load(str(ckpt_path / exp_name / ('opt_' + FLAGS.snapshot)), map_location=dev))
        optimizer.param_groups[0]['lr'] = 2 * FLAGS.learning_rate
        optimizer.param_groups[1]['lr'] = FLAGS.learning_rate

    check_mkdir(str(ckpt_path))
    check_mkdir(str(ckpt_path / exp_name))
    # Save arguments
    with open(os.path.join(str(ckpt_path), exp_name, str(timestamp) + '.txt'), 'w') as argsfile:
        argsfile.write(str(FLAGS) + '\n\n')

    # Train
    train_dataset = CK5AugmentStudyDataset(str(FLAGS.data_dir), "train", augment=aug_level,
                                           tile_size=FLAGS.image_size)
    val_dataset = CK5AugmentStudyDataset(str(FLAGS.data_dir), "val", tile_size=FLAGS.image_size, augment=0)
    test_dataset = CK5AugmentStudyDataset(str(FLAGS.data_dir), "test", tile_size=FLAGS.image_size, augment=0)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, drop_last=False, shuffle=True,
                              num_workers=FLAGS.workers)  # batch norm needs to drop last
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.val_batch_size, drop_last=False, shuffle=False,
                            num_workers=FLAGS.workers)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.val_batch_size, drop_last=False, shuffle=False,
                            num_workers=FLAGS.workers)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)
    scheduler = CyclicLR(optimizer, base_lr=FLAGS.learning_rate / 10, max_lr=FLAGS.learning_rate, step_size=180 * 40,
                         mode='triangular2')

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    for epoch in range(curr_epoch, FLAGS.epochs+1):
        train_stats = train(train_loader, net, criterion, optimizer, epoch, writer, FLAGS.print_freq, scheduler=scheduler)
        if epoch % 10 == 1:  # so it validates at first epoch too
            val_stats = validate(val_loader, net, criterion_val, optimizer, epoch, ckpt_path, writer,
                                best_record, val_imgs_sample_rate=FLAGS.val_imgs_sample_rate,
                                val_save_to_img_file=False)
        # scheduler.step(val_loss)
    if epoch % 10 != 1:
        val_stats = validate(val_loader, net, criterion_val, optimizer, epoch, best_record,
                 val_imgs_sample_rate=FLAGS.val_imgs_sample_rate, val_save_to_img_file=False)
    test_loss, acc, acc_cls, dice, dice_cls = test(test_loader, net, criterion_val)
    test_stats = {'loss': test_loss, 'acc': acc, 'acc_cls': acc_cls, 'dice': dice, 'dice_cls': dice_cls}

    train_msg = 'Train result - val loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}'.format(
        train_stats['loss'], train_stats['acc'], train_stats['acc_cls'], train_stats['dice'], train_stats['dice_cls'])
    val_msg = 'Val result - val loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}'.format(
        val_stats['loss'], val_stats['acc'], val_stats['acc_cls'], val_stats['dice'], val_stats['dice_cls'])
    test_msg = 'Test result - val loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}'.format(
        test_loss, acc, acc_cls, dice, dice_cls)

    return train_stats, val_stats, test_stats


def main(FLAGS):
    global ckpt_path, writer
    ckpt_path, writer = Path(ckpt_path), writer
    check_mkdir(ckpt_path)

    print("Running on GPUs: {}".format(FLAGS.gpu_ids))
    print("Memory: Image size: {}, Batch size: {},  Net filter num: {}".format(FLAGS.image_size, FLAGS.batch_size,
                                                                               FLAGS.num_filters))
    print("Using network {}, {} loss".format(FLAGS.network_id, FLAGS.losstype))
    print("Saving results in {}".format(str(ckpt_path)))
    print("Begin training ...")

    augment_levels = list(range(5))  # augmentation levels are from 0 to 5
    aug_stats_train, aug_stats_val, aug_stats_test = [], [], []
    ckpt_path = ckpt_path / "aug_0"
    for aug in augment_levels:
        print("Training for augmentation level = {}".format(aug))
        ckpt_path = ckpt_path.parent / "aug_{}".format(aug)
        train_stats, val_stats, test_stats = run_aug_exp(FLAGS, aug) # run model
        aug_stats_train.append(train_stats)
        aug_stats_val.append(val_stats)
        aug_stats_test.append(test_stats)


    # But these don't necessarily have to be the results for the best architectures - just the final ones ...
    with open(str(ckpt_path.parent/"inc_aug_result"), 'w') as save_file:
        save_file.write("-- Train --\n")
        for idx, train_stats in enumerate(aug_stats_train):
            save_file.write("Aug level {} - {}\n".format(idx, train_stats))
        save_file.write('\n')

        save_file.write("-- Validate --\n")
        for idx, test_stats in enumerate(aug_stats_val):
            save_file.write("Aug level {} - {}".format(idx, test_stats))
        save_file.write('\n')

        save_file.write("-- Test --\n")
        for idx, test_stats in enumerate(aug_stats_test):
            save_file.write("Aug level {} - {}".format(idx, test_stats))
        save_file.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=299)
    parser.add_argument('--downsample', type=float, default=2.0)
    parser.add_argument('--full_glands', type=str2bool, default="y")

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument('--network_id', type=str, default="inception_v3")
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--learning_rate_patience', default=50, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    #parser.add_argument('--momentum', default=0.95, type=float) #for SGD not ADAM
    parser.add_argument('--losstype', default='ce', choices=['dice', 'ce'])

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--val_batch_size', default=40, type=int)
    parser.add_argument('--val_imgs_sample_rate', default=0.05, type=float)
    parser.add_argument('--snapshot', default='')

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')
    parser.add_argument('--checkpoint_folder', default='', type=str, help=None)

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")

    timestamp = get_time_stamp()
    if on_cluster():
        ckpt_path = "/well/win/users/achatrian/cancer_phenotype/logs/" + "ck5_aug_" + timestamp + "/ckpt"
    else:
        ckpt_path = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Logs"
    exp_name = ''
    writer = SummaryWriter(os.path.join(str(ckpt_path), exp_name + "_" + FLAGS.snapshot))
    visualize = ToTensor()

    main(FLAGS)
