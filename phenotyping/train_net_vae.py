#!/usr/bin/python

import datetime
import os
import random
import argparse
from pathlib import Path
from numbers import Integral

import numpy as np
import torch
from torchvision.transforms import ToTensor  #NB careful -- this changes the range from 0:255 to 0:1 !!!
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from imageio import imwrite
from cv2 import cvtColor, COLOR_GRAY2RGB
#from torchnet.logger import MeterLogger

import warnings

from models import *
from utils import get_time_stamp, check_mkdir, str2bool, evaluate_multilabel, colorize, MultiLabelSoftDiceLoss, AverageMeter

from gland_dataset import GlandDataset

torch.cudnn.benchmark = True

timestamp = get_time_stamp()
ckpt_path = str(Path.home()) + "/ProstateCancer/logs/" + timestamp + "/ckpt"
exp_name = ''
writer = SummaryWriter(os.path.join(ckpt_path, exp_name))
visualize = ToTensor()

FAKE_LABEL=0

def train(train_loader, netG, netD, ce, mse, optimizerG, optimizerD, epoch, load_weightmap, print_freq):
    train_lossD = AverageMeter()
    train_loss_vae = AverageMeter()
    train_lossG = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(tqdm(train_loader)):

        ### TRAIN D WITH REAL DATA###
        if not load_weightmap:
            inputs = data
        else:
            inputs, weightmaps = data
        N = inputs.size(0)
        inputs = Variable(inputs)
        labels = Variable((torch.zeros(N) if not FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
        #TODO check if filling tensors like done in main of dcgan-vae is quicker?
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = label.cuda()
            if load_weightmap:
                weightmaps = weightmaps.cuda()

        optimizerD.zero_grad() #needs to be done at every iteration
        outputsD = netD(inputs)

        lossD = ce(outputsD, labels)
        if load_weightmap:
            scale = (5*weightmaps + torch.ones_like(weightmaps))
            lossD *= scale
            lossD = lossD.mean()

        lossD.backward()
        optimizerD.step()
        train_lossD.update(lossD.data.item(), N)
        D_x = outputsD.data.mean()
        ######

        ### TRAIN D WITH GENERATED DATA ###
        noise = torch.normal(torch.zeros_like(inputs), torch.ones_like(inputs))
        gen_inputs = netG.decoder(noise)
        labels = Variable((torch.zeros(N) if FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
        if torch.cuda.is_available():
            gen_inputs = inputs.cuda()
            label = label.cuda()
            if load_weightmap:
                weightmaps = weightmaps.cuda()

        optimizerD.zero_grad() #needs to be done at every iteration
        outputsD = netD(gen_inputs)

        lossD = ce(outputsD, labels)
        if load_weightmap:
            scale = (5*weightmaps + torch.ones_like(weightmaps))
            lossD *= scale
            lossD = lossD.mean()

        lossD.backward()
        optimizerD.step()
        D_G_z1 = outputsD.data.mean()
        train_lossD.update(lossD.data.item(), N)
        #####

        ###TRAIN G on reconstruction (on content error)#######
        #NB no need for zero grad on G here?
        netG.zero_grad() #neeeded ???
        encoded = netG.encoder(input)
        mu = encoded[0]
        logvar = encoded[1]
        kld_elements = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld_lossG = torch.sum(kld_elements).mul_(-0.5)
        sampled = netG.sampler(encoded)
        rec = netG.decoder(sampled) #reconstructed

        mse_lossG = mse(rec,inputs)
        vae_loss = kld_lossG + mse_lossG #summed to do optimization in one go?
        vae_loss.backward()
        optimizerG.step()
        #####

        ### Train G based on discrimination result (on style error)###
        netG.zero_grad()
        labels = Variable((torch.zeros(N) if not FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
        rec = netG(input) # this tensor is freed from mem at this point
        outputsD = netD(rec)
        lossG = ce(outputsD, labels)
        lossG.backward()
        optimizerG.step()
        D_G_z2 = outputsD.data.mean()
        ####

        if i % print_freq == 0:
            print('[epoch %d][batch s%d] recon loss: %.4f discr loss: %.4f gen loss: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, i, train_loss_vae.avg, train_lossD.avg, train_lossG.avg, D_x, D_G_z1, D_G_z2))

        curr_iter += 1
        writer.add_scalar('discr_loss', train_lossD.avg, curr_iter)
        writer.add_scalar('recon_loss', train_loss_vae.avg, curr_iter)
        writer.add_scalar('gen_loss', train_lossG.avg, curr_iter)

def validate(val_loader, net, criterion, optimizer, epoch, best_record, val_imgs_sample_rate=0.05, val_save_to_img_file=True):
    net.eval() #!!

    val_loss = AverageMeter()
    inputs_smpl, gts_smpl, predictions_smpl = [], [], []
    for vi, data in enumerate(val_loader): #pass over whole validation dataset
        inputs, gts = data
        N = inputs.size(0)
        with torch.no_grad(): #don't track variable history for backprop (to avoid out of memory)
            #NB @pytorch variables and tensors will be merged in the future
            inputs = Variable(inputs).cuda() if torch.cuda.is_available() else Variable(inputs)
            gts = Variable(gts).cuda() if torch.cuda.is_available() else Variable(inputs)
            outputs = net(inputs)

        val_loss.update(criterion(outputs, gts).data.item(), N)

        val_num = int(np.floor(val_imgs_sample_rate*N))
        val_idx = random.sample(list(range(N)), val_num)
        for idx in val_idx:
            inputs_smpl.append(inputs[idx,...].data.cpu().numpy())
            gts_smpl.append(gts[idx,...].data.cpu().numpy())
            predictions_smpl.append(outputs[idx,...].data.cpu().numpy())
    gts_smpl = np.stack(gts_smpl, axis=0)
    predictions_smpl = np.stack(predictions_smpl, axis=0)

    #Compute metrics
    acc, acc_cls, dice, dice_cls = evaluate_multilabel(predictions_smpl, gts_smpl)
    print('epoch: {:d}, val loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}'.format(
        epoch, val_loss.avg, acc, acc_cls, dice, dice_cls))

    if dice > best_record['dice']:
        best_record['val_loss'] = val_loss.avg
        best_record['epoch'] = epoch
        best_record['acc'] = acc
        best_record['acc_cls'] = acc_cls
        best_record['epoch'] = epoch
        best_record['dice'] = dice
        best_record['dice_cls'] = dice_cls
        snapshot_name = 'epoch_.{:d}_loss_{:.5f}_acc_{:.5f}_dice_{:.5f}_lr_{:.10f}'.format(
            epoch, val_loss.avg, acc, dice, optimizer.param_groups[1]['lr'])
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if val_save_to_img_file:
            to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, (input, gt, pred) in enumerate(zip(inputs_smpl, gts_smpl, predictions_smpl)):
            gt_rgb, pred_rgb = colorize(gt), colorize(pred)
            input_rgb = input.transpose(1,2,0) if input.shape[0] == 3 else cvtColor(input.transpose(1,2,0) , COLOR_GRAY2RGB)
            if val_save_to_img_file:
                imwrite(os.path.join(to_save_dir, "{}_input.png".format(idx)), input_rgb)
                imwrite(os.path.join(to_save_dir, "{}_pred.png".format(idx)), pred_rgb)
                imwrite(os.path.join(to_save_dir, "{}_gt.png".format(idx)), gt_rgb)
            val_visual.extend([visualize(input_rgb), visualize(gt_rgb), visualize(pred_rgb)])
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
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
    if isinstance(FLAGS.gpu_ids[0], Integral):
        torch.cuda.set_device(FLAGS.gpu_ids[0])
        num_gpus = len(FLAGS.gpu_ids)
    else:
        torch.cuda.set_device(FLAGS.gpu_ids)
        num_gpus=1

    ###LOAD MODELS####
    netD = _netD(FLAGS.image_size, num_gpus)
    netG = _netG(FLAGS.imageSize,ngpu)
    netD.apply(weights_init) #initialize
    netG.apply(weights_init)
    if FLAGS.netD != '':
        netD.load_state_dict(torch.load(FLAGS.netD))
    if FLAGS.netG != '':
        netG.load_state_dict(torch.load(FLAGS.netG))
    if torch.cuda.is_available:
        netD.cuda()
        netG.make_cuda()

    if not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and torch.cuda.is_available():
        netD = DataParallel(netD, device_ids=FLAGS.gpu_ids).cuda()
        netG = DataParallel(netG, device_ids=FLAGS.gpu_ids).cuda()

    if len(FLAGS.snapshot) == 0:
        curr_epoch = 1
        best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': [0,0,0], 'dice': -.001, 'dice_cls': [-.001,-.001,-.001]}
    else:
        raise NotImplementedError
        print('training resumes from ' + FLAGS.snapshot)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, FLAGS.snapshot)))
        split_snapshot = FLAGS.snapshot.split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'acc': float(split_snapshot[5]),
                               'acc_cls': [ch for ch in split_snapshot[5] if isinstance(ch, Integral)],
                               'dice': float(split_snapshot[9]),
                               'dice_cls': [ch for ch in split_snapshot[11] if isinstance(ch, Integral)]}
    net.train()
    ###################

    ###CRITERIONS####
    reduce = not FLAGS.load_weightmap
    ce = nn.BCELoss()
    mse = nn.MSELoss()
    if torch.cuda.is_available():
        ce = gan_criterion.cuda()
        mse = vae_criterion.cuda()
    ################

    optimizerD = torch.optim.Adam(netD.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, 0.999))
    optimimizerG = torch.optim.Adam(netG.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, 0.999))

    optimimizerD = torch.optim.Adam(netD.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, 0.999))
    optimimizerG = torch.optim.Adam(netG.parameters(), lr=FLAGS.lr, betas=(FLAGS.beta1, 0.999))

    if len(FLAGS.snapshot) > 0:
        ###NOT IMPLEMENTED
        raise NotImplementedError
        optimizerD.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + FLAGS.snapshot)))
        optimizerD.param_groups[0]['lr'] = 2 * FLAGS.learning_rate
        optimizerD.param_groups[1]['lr'] = FLAGS.learning_rate
        optimizerG.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + FLAGS.snapshot)))
        optimizerG.param_groups[0]['lr'] = 2 * FLAGS.learning_rate
        optimizerG.param_groups[1]['lr'] = FLAGS.learning_rate

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    #Save arguments
    with open(os.path.join(ckpt_path, exp_name, str(timestamp) + '.txt'), 'w') as argsfile:
        argsfile.write(str(FLAGS) + '\n\n')

    #Train
    train_dataset = GlandDataset(FLAGS.data_dir, "train", out_size=FLAGS.image_size, down=FLAGS.downsample,
                    num_class=FLAGS.num_class, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.workers)
    val_dataset = GlandDataset(FLAGS.data_dir, "validate", out_size=FLAGS.image_size, down=FLAGS.downsample,
                    num_class=FLAGS.num_class)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.val_batch_size, shuffle=True, num_workers=FLAGS.workers)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Running on GPUs: {}".format(FLAGS.gpu_ids))
    print("Memory: Image size: {}, Batch size: {} (batchnorm: {}),  Net filter num: {}".format(FLAGS.image_size, FLAGS.batch_size, FLAGS.batchnorm, FLAGS.num_filters))
    print("Network has {} parameters".format(params))
    print("Saving results in {}".format(ckpt_path))
    print("Begin training ...")

    for epoch in range(curr_epoch, FLAGS.epochs):
        train(train_loader, netG, netD, ce, mse, optimizerG, optimizerD, epoch, FLAGS.load_weightmap, FLAGS.print_freq)
        #if epoch % 10 == 1: #so it validates at first epoch too
            #val_loss = validate(val_loader, net, criterion_val, optimizer, epoch,
                        #best_record, val_imgs_sample_rate=FLAGS.val_imgs_sample_rate, val_save_to_img_file=True)
        #scheduler.step(val_loss)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=1024, help='the height / width of the input image to network')
    parser.add_argument('--downsample', type=float, default=2.0)
    parser.add_argument('--grayscale', type=str2bool, default=False)

    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--learning_rate_patience', default=50, type=int)
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
    parser.add_argument('--checkpoint_folder', default='checkpoints', type=str, help='checkpoint folder')
    parser.add_argument('--resume', default='', type=str, help='which checkpoint file to resume the training')

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")
    main(FLAGS)
