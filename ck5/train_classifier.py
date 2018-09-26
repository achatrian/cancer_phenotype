#!/usr/bin/python

import datetime
import os, sys
import random
import argparse
from pathlib import Path
from numbers import Integral
import warnings

import numpy as np
from torchvision.transforms import ToTensor  #NB careful -- this changes the range from 0:255 to 0:1 !!!
import torchvision.utils as vutils
from torch import no_grad, cuda
from tensorboardX import SummaryWriter
from torch import optim
from torch import save, load, stack
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from imageio import imwrite
from cv2 import cvtColor, COLOR_GRAY2RGB
from torchvision.models import resnet101

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
    sys.path.append("/gpfs0/users/win/achatrian/ProstateCancer")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/ProstateCancer")

from ck5.ck5_dataset import CK5Dataset
from ck5.ck5_utils import eval_classification
from mymodel.utils import on_cluster, get_time_stamp, check_mkdir, str2bool, colorize, AverageMeter, get_flags
from mymodel.clr import CyclicLR

cudnn.benchmark = True

ckpt_path = ''
writer = None

def train(train_loader, net, criterion, optimizer, epoch, load_weightmap, print_freq, scheduler=None):
    global ckpt_path
    ckpt_path = ckpt_path  # update with global value

    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(tqdm(train_loader)):
        if not load_weightmap:
            inputs, labels = data
            #labels = labels.squeeze()
        else:
            inputs, labels, weightmaps = data
        N = inputs.size(0)
        inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
        labels = Variable(labels).cuda() if cuda.is_available() else Variable(labels)
        if load_weightmap and cuda.is_available():
            weightmaps = weightmaps.cuda()

        optimizer.zero_grad() #needs to be done at every iteration
        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.batch_step()

        train_loss.update(loss.data.item(), N)

        #Compute training metrics:
        predictions = outputs.data.cpu().numpy()
        targets = labels.data.cpu().numpy()
        acc, acc_cls, dice, dice_cls = eval_classification(predictions, targets)

        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)
        writer.add_scalar('train_acc', acc, curr_iter)
        writer.add_scalar('train_dice', dice, curr_iter)
        if i % print_freq == 0:
            tqdm.write('epoch: {}, batch: {} - train - loss: {:.5f}, acc: {:.2f}, acc_cls: {}. dice: {:.2f}, dice_cls: {}'.format(
            epoch, i,  train_loss.avg, acc, acc_cls, dice, dice_cls))

def validate(val_loader, net, criterion, optimizer, epoch, best_record, val_imgs_sample_rate=0.05, val_save_to_img_file=True):
    global ckpt_path, writer
    ckpt_path, writer = ckpt_path, writer  # update with global value

    net.eval() #!!

    val_loss = AverageMeter()
    inputs_smpl, targets_smpl, predictions_smpl = [], [], []
    for vi, data in enumerate(val_loader): #pass over whole validation dataset
        inputs, gts = data
        N = inputs.size(0)
        with no_grad(): #don't track variable history for backprop (to avoid out of memory)
            #NB @pytorch variables and tensors will be merged in the future
            inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
            targets = Variable(gts).cuda() if cuda.is_available() else Variable(inputs)
            outputs = net(inputs)

        val_loss.update(criterion(outputs, gts).data.item(), N)

        val_num = int(np.floor(val_imgs_sample_rate*N))
        val_idx = random.sample(list(range(N)), val_num)
        for idx in val_idx:
            inputs_smpl.append(inputs[idx,...].data.cpu().numpy())
            targets_smpl.append(gts[idx,...].data.cpu().numpy())
            predictions_smpl.append(outputs[idx,...].data.cpu().numpy())
    targets_smpl = np.stack(targets_smpl, axis=0)
    predictions_smpl = np.stack(predictions_smpl, axis=0)

    #Compute metrics
    acc, acc_cls, dice, dice_cls = eval_classification(outputs.cpu().numpy(), targets.cpu().numpy())
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
        save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if val_save_to_img_file:
            to_save_dir = os.path.join(str(ckpt_path), exp_name, str(epoch))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, (input, gt, pred) in enumerate(zip(inputs_smpl, , predictions_smpl)):

            gt_rgb, pred_rgb = colorize(gt), colorize(pred)
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
    return val_loss.avg

def main(FLAGS):
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
    inputs = {'num_classes' : FLAGS.num_class, 'num_channels' : FLAGS.num_filters}

    #Model:
    net = resnet101(pretrained=True)

    if parallel:
        net = DataParallel(net, device_ids=FLAGS.gpu_ids).cuda()

    net.train()

    if not FLAGS.snapshot:
        curr_epoch = 1
        best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': [0,0,0], 'dice': -.001,
                       'dice_cls': [-.001, -.001, -.001], 'lr': FLAGS.learning_rate}
    else:
        print('training resumes from ' + FLAGS.snapshot)

        dev = None if cuda.is_available() else 'cpu'
        state_dict = load(str(ckpt_path/exp_name/FLAGS.snapshot), map_location=dev)
        if not parallel:
            state_dict = {key[7:] : value for key, value in state_dict.items()}
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

    reduce = not FLAGS.load_weightmap
    criterion = BCEWithLogitsLoss(size_average=reduce, reduce=reduce, weight=FLAGS.class_weights)
    #criterion = CrossEntropyLoss(size_average=True, weight=FLAGS.class_weights)
    criterion_val = BCEWithLogitsLoss(size_average=True, reduce=reduce, weight=FLAGS.class_weights)
    #criterion_val = CrossEntropyLoss(size_average=True, weight=FLAGS.class_weights)

    if cuda.is_available(): criterion = criterion.cuda()

    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * FLAGS.learning_rate},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': FLAGS.learning_rate, 'weight_decay': FLAGS.weight_decay}
    ])


    if FLAGS.snapshot:
        optimizer.load_state_dict(load(str(ckpt_path/exp_name/('opt_' + FLAGS.snapshot)), map_location=dev))
        #optimizer.param_groups[0]['lr'] = 2 * FLAGS.learning_rate
        #optimizer.param_groups[1]['lr'] = FLAGS.learning_rate

    check_mkdir(str(ckpt_path))
    check_mkdir(str(ckpt_path/exp_name))
    #Save arguments
    with open(os.path.join(str(ckpt_path), exp_name, str(timestamp) + '.txt'), 'w') as argsfile:
        argsfile.write(str(FLAGS) + '\n\n')

    #Train
    #train_dataset = ProstateDataset(FLAGS.data_dir, "train", out_size=FLAGS.image_size, down=FLAGS.downsample,
    #                num_class=FLAGS.num_class, grayscale=FLAGS.grayscale, augment=True, load_wm=FLAGS.load_weightmap)
    train_dataset = CK5Dataset(FLAGS.data_dir, "train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.workers)
    val_dataset = CK5Dataset(FLAGS.data_dir, "val", augment=False)
    #val_dataset = ProstateDataset(FLAGS.data_dir, "validate", out_size=FLAGS.image_size, down=FLAGS.downsample,
    #               num_class=FLAGS.num_class, grayscale=FLAGS.grayscale)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.val_batch_size, shuffle=True, num_workers=FLAGS.workers)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)
    scheduler = CyclicLR(optimizer, base_lr=FLAGS.learning_rate/10, max_lr=FLAGS.learning_rate, step_size=180*40, mode='triangular2')

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Running on GPUs: {}".format(FLAGS.gpu_ids))
    print("Memory: Image size: {}, Batch size: {},  Net filter num: {}".format(FLAGS.image_size, FLAGS.batch_size, FLAGS.num_filters))
    print("Network has {} parameters".format(params))
    print("Using network {}, {} loss".format(FLAGS.network_id, FLAGS.losstype))
    print("Saving results in {}".format(str(ckpt_path)))
    print("Begin training ...")

    for epoch in range(curr_epoch, FLAGS.epochs):
        train(train_loader, net, criterion, optimizer, epoch, FLAGS.load_weightmap, FLAGS.print_freq, scheduler=scheduler)
        if epoch % 10 == 1: #so it validates at first epoch too
            val_loss = validate(val_loader, net, criterion_val, optimizer, epoch,
                        best_record, val_imgs_sample_rate=FLAGS.val_imgs_sample_rate, val_save_to_img_file=True)
        #scheduler.step(val_loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=512, help='the height / width of the input image to network')
    parser.add_argument('--downsample', type=float, default=2.0)

    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--network_id', type=str, default="resnet101")
    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--learning_rate_patience', default=50, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    #parser.add_argument('--momentum', default=0.95, type=float) #for SGD not ADAM
    parser.add_argument('--losstype', default='ce', choices=['dice', 'ce'])
    parser.add_argument('--class_weights', default=None)
    parser.add_argument('--load_weightmap', type=str2bool, default=False)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--val_batch_size', default=64, type=int)
    parser.add_argument('--val_imgs_sample_rate', default=0.05, type=float)
    parser.add_argument('--snapshot', default='')

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/ProstateCancer/Dataset")
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')
    parser.add_argument('--checkpoint_folder', default='', type=str, help='checkpoint folder')

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")

    timestamp = get_time_stamp()
    if on_cluster():
        ckpt_path = "/well/win/users/achatrian/ProstateCancer/logs/" + timestamp + "/ckpt"
    else:
        ckpt_path = "/Users/andreachatrian/Documents/Repositories/ProstateCancer/Logs"
    exp_name = ''
    writer = SummaryWriter(os.path.join(str(ckpt_path), exp_name if not FLAGS.snapshot else exp_name + "_" + FLAGS.snapshot))
    visualize = ToTensor()

    main(FLAGS)
