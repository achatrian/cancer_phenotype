#!/usr/bin/python

import datetime
import os, sys
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
from torch.nn import DataParallel, BCELoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from imageio import imwrite
from cv2 import cvtColor, COLOR_GRAY2RGB
#from torchnet.logger import MeterLogger

import warnings

from gland_dataset import GlandDataset

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
    sys.path.append(os.path.expanduser('~') + 'ProstateCancer/mymodels')
else:
    sys.path.append(os.path.expanduser('~') + '/Documents/Repositories/ProstateCancer/mymodels')
from models import _netD, _netG, weights_init
from utils import get_time_stamp, check_mkdir, str2bool, evaluate_multilabel, colorize, MultiLabelSoftDiceLoss, AverageMeter

cudnn.benchmark = True

timestamp = get_time_stamp()
ckpt_path = "/well/win/users/achatrian/ProstateCancer/logs/" + "vae" + timestamp + "/ckpt"
exp_name = ''
writer = SummaryWriter(os.path.join(ckpt_path, exp_name))
visualize = ToTensor()
FAKE_LABEL=0

def train(train_loader, netG, netD, nz, ce, mse, optimizerG, optimizerD, epoch, load_weightmap, print_freq):
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
            labels = labels.cuda()
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
        noise = torch.normal(torch.zeros((N, nz // (4*4), 4, 4)), torch.ones((N, nz // (4*4), 4, 4))) #feeding a random latent vector to decoder # does this encourage repr for glands away from z=0 ???
        if torch.cuda.is_available():
            noise = noise.cuda()
        gen_inputs = netG.decoder(noise)
        labels = Variable((torch.zeros(N) if FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
        if torch.cuda.is_available():
            gen_inputs = inputs.cuda()
            labels = labels.cuda()
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
        #####

        ###TRAIN G on reconstruction (on content error)#######
        #NB no need for zero grad on G here?
        netG.zero_grad() #neeeded ???
        encoded = netG.encoder(inputs)
        mu = encoded[0]
        logvar = encoded[1]
        kld_elements = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld_lossG = torch.sum(kld_elements).mul_(-0.5)
        sampled = netG.sampler(encoded)
        rec = netG.decoder(sampled) #reconstructed

        mse_lossG = mse(rec, inputs)
        vae_loss = kld_lossG + mse_lossG #summed to do optimization in one go?
        vae_loss.backward()
        optimizerG.step()
        train_loss_vae.update(vae_loss.data.item(), N)
        #####

        ### Train G based on discrimination result (on style error)###
        netG.zero_grad()
        labels = Variable((torch.zeros(N) if not FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
        if torch.cuda.is_available():
            labels = labels.cuda()
        rec = netG(inputs) # this tensor is freed from mem at this point
        outputsD = netD(rec)
        lossG = ce(outputsD, labels)
        lossG.backward()
        optimizerG.step() #only changes parameters on G network (even though error is also backpropagated through D)
        D_G_z2 = outputsD.data.mean()
        train_lossG.update(lossG.data.item(), N)
        ####

        if i % print_freq == 0:
            tqdm.write('[epoch %d][batch %d] recon loss: %.4f discr loss: %.4f gen loss: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, i, train_loss_vae.avg, train_lossD.avg, train_lossG.avg, D_x, D_G_z1, D_G_z2))

        curr_iter += 1
        writer.add_scalar('discr_loss', train_lossD.avg, curr_iter)
        writer.add_scalar('recon_loss', train_loss_vae.avg, curr_iter)
        writer.add_scalar('gen_loss', train_lossG.avg, curr_iter)

def validate(val_loader, netD, netG, ce, mse, optimizerD, optimizerG, epoch, best_record,
             val_imgs_sample_rate=0.1, val_save_to_img_file=True):
    netD.eval() # !!
    netG.eval()  # !!

    val_lossD = AverageMeter()
    val_loss_vae = AverageMeter()
    val_lossG = AverageMeter()
    inputs_smpl, reconstructions_smpl = [], []
    for vi, data in enumerate(val_loader): #pass over whole validation dataset
        inputs = data
        N = inputs.size(0)
        #Validate discrimination by D net
        with torch.no_grad(): #don't track variable history for backprop (to avoid out of memory)
            #NB @pytorch variables and tensors will be merged in the future
            inputs = Variable(inputs)
            labels = Variable((torch.zeros(N) if not FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputsD = netD(inputs)

        val_lossD.update(ce(outputsD, labels).data.item(), N)

        # Validate generation G
        encoded = netG.encoder(inputs)
        with torch.no_grad():
            mu = encoded[0]
            logvar = encoded[1]
            sampled = netG.sampler(encoded)
            rec = netG.decoder(sampled) #reconstructed

            kld_elements = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            kld_lossG = torch.sum(kld_elements).mul_(-0.5)
            mse_lossG = mse(rec, inputs)
            vae_loss = kld_lossG + mse_lossG
            val_loss_vae.update(vae_loss.data.item(), N)

        #Validate G on creating good repr (style)
        with torch.no_grad():
            labels = Variable((torch.zeros(N) if not FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
            if torch.cuda.is_available():
                labels = labels.cuda()
            rec = netG(inputs)  # this tensor is freed from mem at this point
            outputsD = netD(rec)
            lossG = ce(outputsD, labels)
            D_G_z2 = outputsD.data.mean()
            val_lossG.update(lossG.data.item(), N)

        #Save some reconstruction examples
        val_num = int(np.floor(val_imgs_sample_rate*N))
        val_idx = random.sample(list(range(N)), val_num)
        for idx in val_idx:
            inputs_smpl.append(inputs[idx,...].data.cpu().numpy())
            reconstructions_smpl.append(rec[idx,...].data.cpu().numpy())

    if val_loss_vae.avg < best_record['loss_vae']:
        best_record['loss_vae'] = val_loss_vae.avg
        best_record['loss_G'] = val_lossG.avg
        best_record['loss_D'] = val_lossD.avg
        best_record['epoch'] = epoch

        snapshot_name = 'epoch_.{:d}_vae_loss_{:.5f}_loss_G_{:.5f}_loss_D_{:.5f}'.format(
            epoch, val_loss_vae.avg, val_lossG.avg, val_lossD.avg)
        torch.save(netG.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + 'G.pth'))
        torch.save(optimizerG.state_dict(), os.path.join(ckpt_path, exp_name, 'optG_' + snapshot_name + '.pth'))
        torch.save(netD.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + 'D.pth'))
        torch.save(optimizerD.state_dict(), os.path.join(ckpt_path, exp_name, 'optD_' + snapshot_name + '.pth'))

        if val_save_to_img_file:
            to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, (input, rec) in enumerate(zip(inputs_smpl, reconstructions_smpl)):
            input_rgb = input.transpose(1,2,0) if input.shape[0] == 3 else cvtColor(input.transpose(1,2,0) , COLOR_GRAY2RGB)
            rec_rgb = rec.transpose(1,2,0) if input.shape[0] == 3 else cvtColor(rec.transpose(1,2,0) , COLOR_GRAY2RGB)
            if val_save_to_img_file:
                imwrite(os.path.join(to_save_dir, "{}_input.png".format(idx)), input_rgb)
                imwrite(os.path.join(to_save_dir, "{}_rec.png".format(idx)), rec_rgb)
            val_visual.extend([visualize(input_rgb), visualize(rec_rgb)])
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=2, padding=5)
        writer.add_image(snapshot_name, val_visual)

        print('-----------------------------------------------------------------------------------------------------------')
        print("best val record (epoch{:d}):, vae_loss: {:.5f}, loss_G {:.5f}, loss_D: {:.5f}".format(
            best_record['epoch'], best_record['loss_vae'], best_record['loss_G'], best_record['loss_D']))
        print('-----------------------------------------------------------------------------------------------------------')
    writer.add_scalar('val_vae_loss', val_loss_vae.avg, epoch)
    writer.add_scalar('val_loss_G', val_lossG.avg, epoch)
    writer.add_scalar('val_loss_D', val_lossD.avg, epoch)
    #writer.add_scalar('lr', optimizerG.param_groups[1]['lr'], epoch)

    netG.train() #reset network state to train (e.g. for batch norm)
    netD.train()
    return val_loss_vae.avg, val_lossD.avg

def main(FLAGS):
    if torch.cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral):
            torch.cuda.set_device(FLAGS.gpu_ids)
            num_gpus=1
        else:
            torch.cuda.set_device(FLAGS.gpu_ids[0])
            num_gpus = len(FLAGS.gpu_ids)
    else:
        num_gpus=0

    ###LOAD MODELS###
    num_channels = 1 if FLAGS.grayscale else 3
    netD = _netD(FLAGS.image_size, num_gpus, FLAGS.num_filt_discr, num_channels, batchnorm=FLAGS.batchnorm) # (self, image_size, ngpu, num_filt_discr, num_channels=3):
    netG = _netG(FLAGS.image_size, num_gpus, FLAGS.num_filt_gen, FLAGS.num_lat_dim, num_channels, batchnorm=FLAGS.batchnorm) # (self, image_size, ngpu, num_filt_gen, num_lat_dim, num_channels=3):
    netD.apply(weights_init) #initialize
    netG.apply(weights_init)
    #if FLAGS.netD != '':
     #   netD.load_state_dict(torch.load(FLAGS.netD))
    #if FLAGS.netG != '':
    #    netG.load_state_dict(torch.load(FLAGS.netG))
    if torch.cuda.is_available():
        netD.cuda()
        netG.cuda()

    if len(FLAGS.snapshot) == 0:
        curr_epoch = 1
        best_record = {'epoch': 0, 'loss_vae': 1e10, 'loss_G': 1e10, 'loss_D': 1e10}
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
    netD.train()
    netG.train()

    #Check models structure:
    #print(netD)
    #print(netG)
    ###################

    ###CRITERIONS####
    reduce = not FLAGS.load_weightmap
    ce = BCELoss(size_average=True)
    mse = MSELoss(size_average=True)
    if torch.cuda.is_available():
        ce = ce.cuda()
        mse = mse.cuda()
    #################

    optimizerD = torch.optim.Adam(netD.parameters(), lr=FLAGS.learning_rate)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=FLAGS.learning_rate)

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
    train_dataset = GlandDataset(FLAGS.data_dir, "train", augment=True)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.workers)
    val_dataset = GlandDataset(FLAGS.data_dir, "validate")
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.val_batch_size, shuffle=True, num_workers=FLAGS.workers)
    schedulerG = ReduceLROnPlateau(optimizerG, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)
    schedulerD = ReduceLROnPlateau(optimizerD, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)

    model_parametersD = filter(lambda p: p.requires_grad, netD.parameters())
    model_parametersG = filter(lambda p: p.requires_grad, netG.parameters())
    paramsD = sum([np.prod(p.size()) for p in model_parametersD])
    paramsG = sum([np.prod(p.size()) for p in model_parametersG])
    print("Running on GPUs: {}".format(FLAGS.gpu_ids))
    print("Memory: Image size: {}, Batch size: {}".format(FLAGS.image_size, FLAGS.batch_size))
    print("Hidden dim: {}, NetD filter num: {}, NetG filter num: {}".format(FLAGS.num_lat_dim, FLAGS.num_filt_discr, FLAGS.num_filt_gen))
    print("Discriminator: {} parameters; Autoencoder: {} parameters".format(paramsD, paramsG))
    print("Saving results in {}".format(ckpt_path))
    print("Begin training ...")

    for epoch in range(curr_epoch, FLAGS.epochs):
        train(train_loader, netG, netD, FLAGS.num_lat_dim, ce, mse, optimizerG, optimizerD, epoch, FLAGS.load_weightmap, FLAGS.print_freq)
        if epoch % 20 == 1: #so it validates at first epoch too
            val_loss_vae, val_lossD = validate(val_loader, netD, netG, ce, mse, optimizerD, optimizerG, epoch, best_record,
                                val_imgs_sample_rate=FLAGS.val_imgs_sample_rate, val_save_to_img_file=True)
            tqdm.write("Epoch {} val_loss_vae {} val_lossD {}".format(epoch, val_loss_vae, val_lossD))
        schedulerG.step(val_loss_vae)
        schedulerD.step(val_lossD)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--grayscale', type=str2bool, default=False)

    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batchnorm', type=str2bool, default='y')

    parser.add_argument('-ndf', '--num_filt_discr', type=int, default=35, help='mcd number of filters for unet conv layers')
    parser.add_argument('-ngf', '--num_filt_gen', type=int, default=35, help='mcd number of filters for unet conv layers')
    parser.add_argument('-nz', '--num_lat_dim', type=int, default=1024, help='size of the latent z vector')

    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--learning_rate_patience', default=50, type=int)
    #parser.add_argument('--momentum', default=0.95, type=float) #for SGD not ADAM
    parser.add_argument('--class_weights', default=None)
    parser.add_argument('--load_weightmap', type=str2bool, default=False)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--val_batch_size', default=45, type=int)
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
