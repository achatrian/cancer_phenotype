#!/usr/bin/python

import os, sys
import random
import argparse
from numbers import Integral

import numpy as np
import torch
from torchvision.transforms import ToTensor  #NB careful -- this changes the range from 0:255 to 0:1 !!!
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel, L1Loss, BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from imageio import imwrite
from cv2 import cvtColor, COLOR_GRAY2RGB
#from torchnet.logger import MeterLogger

import warnings

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

from repo_old.phenotyping import GlandPatchDataset, GlandDataset
from repo_old.phenotyping.models import VAE, Discriminator, weights_init
from segment.utils import get_time_stamp, check_mkdir, str2bool, AverageMeter
from repo_old.clr import CyclicLR

cudnn.benchmark = True

timestamp = get_time_stamp()
ckpt_path = "/well/rittscher/users/achatrian/cancer_phenotype/logs/" + "vae" + timestamp + "/ckpt" if on_cluster() \
                            else os.path.expanduser('~') + "/cancer_phenotype/Logs/" + "vae" + timestamp + "/ckpt"
exp_name = ''
writer = SummaryWriter(os.path.join(ckpt_path, exp_name))
visualize = ToTensor()
FAKE_LABEL = 0

equilibrium = 0.5
margin = 0.3


def train(train_loader, vae, discr, nz, l1, opt_enc, opt_dec, opt_dis, opt_vae, schedulers, gamma=1.0, epoch=1, load_weightmap=False, print_freq=10):
    train_loss_enc = AverageMeter()
    train_loss_dec = AverageMeter()
    train_loss_gan = AverageMeter()
    train_loss_rec = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    global margin
    margin *= 0.99
    # if epoch < 20:
    #     gaussian_filter = get_gaussian_blur(epoch, alpha=0.1)
    l1_rec = L1Loss(size_average=False, reduce=False)
    ce = BCELoss(size_average=True)
    for i, data in enumerate(tqdm(train_loader)):

        inputs = data[0]
        gts = data[1]
        gts = torch.clamp(gts, 1)
        gts_weight = gts * 4 + 1.0  # turn to 0-1 first, then scale and use as weightmap
        N, y, x = inputs.size(0), inputs.size(2), inputs.size(3)

        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            gts_weight = gts_weight.cuda()

        encoded, reconstructed, generated = vae(inputs)  # generated does not depend on inputs

        mu = encoded[0]
        logvar = encoded[1] # !!! log of sigma^2
        # Prior loss (KL div between recognition model and prior)#######
        kld_elements = (mu.pow(2) + logvar.exp() - 1 - logvar) / 2  #log of variance learned by network (why??)
        kld_loss = torch.clamp(torch.mean(kld_elements), max=1e15)

        # l1 loss between discriminator layers for reconstructed and real input
        l_rec0, l_rec1 = discr(reconstructed, output_layer=True)
        l_real0, l_real1 = discr(inputs, output_layer=True)
        l_real0 = Variable(l_real0, requires_grad=False)
        l_real1 = Variable(l_real1, requires_grad=False)
        l1_loss = torch.clamp(l1(l_rec0, l_real0), max=1e15) + torch.clamp(l1(l_rec1, l_real1), max=1e15)

        # GAN loss:
        d_x = discr(inputs)
        d_g_z = discr(reconstructed)
        d_g_zp = discr(generated)
        # Smooth labels
        outputs_dis = torch.cat((d_x, d_g_z, d_g_zp), dim=0)
        targets = torch.cat((torch.rand(d_x.shape[0]) * 0.19 + 0.8,
                             torch.rand(d_g_z.shape[0] + d_g_zp.shape[0]) * 0.2), dim=0)
        if torch.cuda.is_available():
            outputs_dis = outputs_dis.cuda()
            targets = targets.cuda()
        dis_loss = ce(outputs_dis, targets)
        gen_loss = - dis_loss

        # Disable training of either decoder or discriminator if optimization becomes unbalanced
        # (as measured by comparing to some predefined bounds) (as in https://github.com/lucabergamini/VAEGAN-PYTORCH/blob/master/main.py)

        if torch.mean(d_x).data.item() > equilibrium + margin and \
            torch.mean(d_g_z) < equilibrium - margin and torch.mean(d_g_zp) < equilibrium + margin:
            train_dis = False
        else:
            train_dis = True

        if torch.mean(d_x).data.item() < equilibrium - margin and \
                torch.mean(d_g_z) > equilibrium + margin and torch.mean(d_g_zp) > equilibrium + margin:
            train_dec = False
        else:
            train_dec = True

        #Optimize:
        opt_enc.zero_grad()
        loss_enc = kld_loss + l1_loss
        loss_enc.backward(retain_graph=True)  # since l1 loss is used again in decoder optimization
        opt_enc.step()  # encoder
        train_loss_enc.update(loss_enc.data.item(), N)
        #del kld_loss  # release graph on unneeded loss

        loss_dec = gamma * l1_loss + gen_loss
        train_loss_dec.update(loss_dec.data.item(), N)
        if train_dec:
            opt_dec.zero_grad()
            loss_dec.backward(retain_graph=True)   # since gan loss is used again
            opt_dec.step()  # decoder

            #del loss_enc, l1_loss  # release graph on unneeded loss

        if train_dis:
            opt_dis.zero_grad()
            dis_loss.backward(retain_graph=True)
            opt_dis.step()  # discriminator
        train_loss_gan.update(dis_loss.data.item(), N)

        #if train_dec: del loss_dec
        #del dis_loss, generated, l_rec


        # Optimize encoder w.r. to l1 loss for reconstruction (calculated later to save gpu space)

        opt_vae.zero_grad()
        rec_loss = torch.sum(l1_rec(reconstructed, inputs) * gts_weight)
        vae_loss = rec_loss + l1_loss  # I added l1 loss here ...
        vae_loss.backward()
        opt_vae.step()
        train_loss_rec.update(rec_loss.data.item(), N)


        schedulers['enc'].batch_step()
        schedulers['dec'].batch_step()
        schedulers['dis'].batch_step()
        schedulers['vae'].batch_step()

        if i % print_freq == 0:
            tqdm.write('[epoch %d][batch %d] losses: enc: %.4E, dec: %.4E, gan: %.4E, rec %.4E | D(x): %.4f D(G(z)): %.4f / %.4f'
                % (epoch, i, train_loss_enc.avg, train_loss_dec.avg, train_loss_gan.avg, train_loss_rec.avg,
                   d_x.mean().item(), d_g_z.mean().data.item(), d_g_zp.mean().data.item()))


        #del d_x, d_g_z, d_g_zp

        curr_iter += 1
        writer.add_scalar('enc_loss', train_loss_enc.avg, curr_iter)
        writer.add_scalar('dec_loss', train_loss_dec.avg, curr_iter)
        writer.add_scalar('dis_loss', train_loss_gan.avg, curr_iter)

def validate(val_loader, discr, vae, nz, l1, optimizers, gamma=1, epoch=1, best_record=None,
             val_imgs_sample_rate=0.1, val_save_to_img_file=True):
    discr.eval() # !!
    vae.eval()  # !!

    val_loss_rec = AverageMeter()
    val_loss_enc = AverageMeter()
    val_loss_dec = AverageMeter()
    val_loss_gan = AverageMeter()
    inputs_smpl, reconstructions_smpl = [], []
    l1_rec = L1Loss(size_average=False, reduce=False)
    for vi, data in enumerate(val_loader): #pass over whole validation dataset
        inputs = data[0]
        gts = data[1]
        gts = torch.clamp(gts, 1) * 4 + 1.0  # turn to 0-1 first, then scale and use as weightmap
        N = inputs.size(0)
        #Validate discrimination by D net
        with torch.no_grad(): #don't track variable history for backprop (to avoid out of memory)
            #NB @pytorch variables and tensors will be merged in the future
            inputs = Variable(inputs)
            labels = Variable((torch.zeros(N) if not FAKE_LABEL else torch.ones(N)).type(torch.FloatTensor))
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                gts = gts.cuda()
                labels = labels.cuda()

            encoded, reconstructed, generated = vae(inputs)  # generated does not depend on inputs

            mu = encoded[0]
            logvar = encoded[1]  # !!! log of sigma^2
            ###Prior loss (KL div between recognition model and prior)#######
            kld_elements = (mu.pow(2) + logvar.exp() - 1 - logvar) / 2  # log of variance learned by network (why??)
            kld_loss = - torch.mean(kld_elements)

            # l1 loss between discriminator layers for reconstructed and real input
            l_rec0, l_rec1 = discr(reconstructed, output_layer=True)
            l_real0, l_real1 = discr(inputs, output_layer=True)
            l_real0 = Variable(l_real0, requires_grad=False)
            l_real1 = Variable(l_real1, requires_grad=False)
            l1_loss = torch.clamp(l1(l_rec0, l_real0), max=1e15) + torch.clamp(l1(l_rec1, l_real1), max=1e15)

            # GAN loss:
            d_x = discr(inputs)
            d_g_z = discr(reconstructed)
            d_g_zp = discr(generated)
            dis_loss = - torch.mean(torch.log(d_x + 1e-3) + torch.log(1 - d_g_z + 1e-3) +
                                    torch.log(1 - d_g_zp + 1e-3))
            gen_loss = torch.mean(torch.log(1 - d_g_z + 1e-3) + torch.log(1 - d_g_zp + 1e-3))

            # Losses:
            loss_enc = kld_loss + l1_loss
            loss_dec = gamma * l1_loss + gen_loss
            loss_rec = l1_rec(reconstructed, inputs) * gts  # used scaled ground - truth as weightmap
            loss_rec = loss_rec.sum()
            val_loss_enc.update(loss_enc.data.item(), N)
            val_loss_dec.update(loss_dec.data.item(), N)
            val_loss_gan.update(dis_loss.data.item(), N)
            val_loss_rec.update(loss_rec.data.item(), N)


        #Save some reconstruction examples
        val_num = max(int(np.floor(val_imgs_sample_rate*N)), 1)
        val_idx = random.sample(list(range(N)), val_num)
        for idx in val_idx:
            inputs_smpl.append(inputs[idx, ...].data.cpu().numpy())
            reconstructions_smpl.append(reconstructed[idx,...].data.cpu().numpy())

    if val_loss_rec.avg < best_record['loss_rec'] or val_loss_enc.avg < best_record['loss_enc']:
        best_record['loss_enc'] = val_loss_enc.avg
        best_record['loss_dec'] = val_loss_dec.avg
        best_record['loss_gan'] = val_loss_gan.avg
        best_record['loss_rec'] = val_loss_rec.avg
        best_record['epoch'] = epoch

        snapshot_name = 'epoch_.{:d}_rec_loss_{:.5f}_enc_loss_{:.5f}_dec_loss_{:.5f}_dis_loss_{:.5f}'.format(
            epoch, val_loss_rec.avg, val_loss_enc.avg, val_loss_dec.avg, val_loss_gan.avg)
        torch.save(vae.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + 'G.pth'))
        torch.save(discr.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + 'D.pth'))
        torch.save(optimizers['enc'].state_dict(), os.path.join(ckpt_path, exp_name, 'opt_enc_' + snapshot_name + '.pth'))
        torch.save(optimizers['dec'].state_dict(), os.path.join(ckpt_path, exp_name, 'opt_dec_' + snapshot_name + '.pth'))
        torch.save(optimizers['dis'].state_dict(), os.path.join(ckpt_path, exp_name, 'opt_dis_' + snapshot_name + '.pth'))

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
        print("best val record (epoch{:d}):, loss_rec: {:.5f}, loss_enc {:.5f}, loss_dec: {:.5f}, loss_gan: {:.5f}".format(
            best_record['epoch'], best_record['loss_rec'], best_record['loss_enc'], best_record['loss_dec'], best_record['loss_gan']))
        print('-----------------------------------------------------------------------------------------------------------')
    writer.add_scalar('val_rec_loss', val_loss_rec.avg, epoch)
    writer.add_scalar('val_enc_loss', val_loss_enc.avg, epoch)
    writer.add_scalar('val_dec_loss', val_loss_dec.avg, epoch)
    writer.add_scalar('val_dis_loss', val_loss_gan.avg, epoch)
    #writer.add_scalar('lr', optimizerG.param_groups[1]['lr'], epoch)

    vae.train() #reset network state to train (e.g. for batch norm)
    discr.train()
    return val_loss_rec.avg, val_loss_enc.avg, val_loss_dec.avg, val_loss_gan.avg

def main(FLAGS):

    # For data parallel, default gpu must be set as first available one in node (pytorch rule)
    if torch.cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral):
            torch.cuda.set_device(FLAGS.gpu_ids)
            num_gpus=1
        else:
            torch.cuda.set_device(FLAGS.gpu_ids[0])
            num_gpus = len(FLAGS.gpu_ids)
            FLAGS.batch_size -= FLAGS.batch_size % len(FLAGS.gpu_ids)  # ensure that there are enough examples per GPU
            FLAGS.val_batch_size -= FLAGS.batch_size % len(FLAGS.gpu_ids)  # " "
    else:
        num_gpus=0

    ###LOAD MODELS###
    num_channels = 3
    discr = Discriminator(FLAGS.image_size, num_gpus, FLAGS.num_filt_discr, num_channels) # (self, image_size, ngpu, num_filt_discr, num_channels=3):
    vae = VAE(FLAGS.image_size, num_gpus, FLAGS.num_filt_gen, FLAGS.num_lat_dim) # (self, image_size, ngpu, num_filt_gen, num_lat_dim, num_channels=3):
    discr.apply(weights_init) #initialize
    vae.apply(weights_init)

    parallel = not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and torch.cuda.is_available()
    if parallel:
        discr = DataParallel(discr, device_ids=FLAGS.gpu_ids).cuda(device=torch.device('cuda:{}'.format(FLAGS.gpu_ids[0])))
        vae = DataParallel(vae, device_ids=FLAGS.gpu_ids).cuda(device=torch.device('cuda:{}'.format(FLAGS.gpu_ids[1])))

    #if FLAGS.discr != '':
     #   discr.load_state_dict(torch.load(FLAGS.discr))
    #if FLAGS.vae != '':
    #    vae.load_state_dict(torch.load(FLAGS.vae))
    if torch.cuda.is_available():
        discr.cuda()
        vae.cuda()

    if len(FLAGS.snapshot) == 0:
        curr_epoch = 1
        best_record = {'epoch': 0, 'loss_rec': 1e10, 'loss_enc': 1e10, 'loss_dec': 1e10, 'loss_gan': 1e10}
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
    discr.train()
    vae.train()

    #Check models structure:
    #print(discr)
    #print(vae)
    ###################

    ###CRITERIONS####
    reduce = not FLAGS.load_weightmap
    l1 = L1Loss(size_average=True)
    if torch.cuda.is_available():
        l1 = l1.cuda()
    #################

    if parallel:
        opt_dis = torch.optim.Adam(discr.module.parameters(), lr=FLAGS.learning_rate)
        opt_enc = torch.optim.Adam(vae.module.encoder.parameters(), lr=FLAGS.learning_rate)
        opt_dec = torch.optim.Adam(vae.module.decoder.parameters(), lr=FLAGS.learning_rate)
    else:
        opt_dis = torch.optim.Adam(discr.parameters(), lr=FLAGS.learning_rate)
        opt_enc = torch.optim.Adam(vae.encoder.parameters(), lr=FLAGS.learning_rate)
        opt_dec = torch.optim.Adam(vae.decoder.parameters(), lr=FLAGS.learning_rate)
    opt_vae = torch.optim.Adam(vae.parameters(), lr=FLAGS.learning_rate)

    if len(FLAGS.snapshot) > 0:
        ###NOT IMPLEMENTED
        # TODO decide whether to incorporate or not
        raise NotImplementedError
        opt_dis.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_dis' + FLAGS.snapshot)))
        opt_dis.param_groups[0]['lr'] = 2 * FLAGS.learning_rate
        opt_dis.param_groups[1]['lr'] = FLAGS.learning_rate
        opt_enc.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_enc' + FLAGS.snapshot)))
        opt_enc.param_groups[0]['lr'] = 2 * FLAGS.learning_rate
        opt_enc.param_groups[1]['lr'] = FLAGS.learning_rate
        opt_dec.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_dec' + FLAGS.snapshot)))
        opt_dec.param_groups[0]['lr'] = 2 * FLAGS.learning_rate
        opt_dec.param_groups[1]['lr'] = FLAGS.learning_rate


    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    #Save arguments
    with open(os.path.join(ckpt_path, exp_name, str(timestamp) + '.txt'), 'w') as argsfile:
        argsfile.write(str(FLAGS) + '\n\n')

    # Train
    if FLAGS.full_glands:
        train_dataset = GlandDataset(FLAGS.data_dir, "train", tile_size=FLAGS.image_size, augment=True, blur=True)
        val_dataset = GlandDataset(FLAGS.data_dir, "val", tile_size=FLAGS.image_size, blur=True)
    else:
        train_dataset = GlandPatchDataset(FLAGS.data_dir, "train", tile_size=FLAGS.image_size, augment=True)
        val_dataset = GlandPatchDataset(FLAGS.data_dir, "val", tile_size=FLAGS.image_size)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, drop_last=True, shuffle=True, num_workers=FLAGS.workers)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.val_batch_size, drop_last=True, shuffle=True, num_workers=FLAGS.workers)
    #scheduler_enc = ReduceLROnPlateau(opt_enc, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)
    #scheduler_dec = ReduceLROnPlateau(opt_dec, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)
    #scheduler_discr = ReduceLROnPlateau(opt_dis, 'min', patience=FLAGS.learning_rate_patience, min_lr=1e-10)
    sche_enc = CyclicLR(opt_enc, base_lr=FLAGS.learning_rate/10, max_lr=FLAGS.learning_rate, step_size=1200, mode ='triangular')
    sche_dec = CyclicLR(opt_dec, base_lr=FLAGS.learning_rate/10, max_lr=FLAGS.learning_rate, step_size=1200, mode ='triangular')
    sche_dis = CyclicLR(opt_dis, base_lr=FLAGS.learning_rate/10, max_lr=FLAGS.learning_rate, step_size=1200, mode ='triangular')
    sche_vae = CyclicLR(opt_vae, base_lr=FLAGS.learning_rate / 10, max_lr=FLAGS.learning_rate, step_size=1200,
                        mode='triangular')


    model_parameters_discr = filter(lambda p: p.requires_grad, discr.parameters())
    model_parameters_vae = filter(lambda p: p.requires_grad, vae.parameters())
    params_discr = sum([np.prod(p.size()) for p in model_parameters_discr])
    params_vae = sum([np.prod(p.size()) for p in model_parameters_vae])
    #if parallel:
    #   module_devs = {'discr' : }
    #print("Running on GPUs: {} - discr on cuda{}, vae on cuda:{}, l1 on cuda:{} :".format(
    #        # FLAGS.gpu_ids, discr.parameters().get_device().index, vae.get_device().index, l1.get_device().index))
    print("Running on GPUs: {}".format(FLAGS.gpu_ids))
    print("Memory: Image size: {}, Batch size: {}".format(FLAGS.image_size, FLAGS.batch_size))
    print("Hidden dim: {}, discr filter num: {}, vae filter num: {}".format(FLAGS.num_lat_dim, FLAGS.num_filt_discr, FLAGS.num_filt_gen))
    print("Discriminator: {} parameters; Autoencoder: {} parameters".format(params_discr, params_vae))
    print("Saving results in {}".format(ckpt_path))
    print("Begin training ...")

    for epoch in range(curr_epoch, FLAGS.epochs):
        train(train_loader, vae, discr, FLAGS.num_lat_dim, l1, opt_enc, opt_dec, opt_dis, opt_vae,
              schedulers={'enc': sche_enc, 'dec': sche_dec, 'dis': sche_dis, 'vae' : sche_vae},
              epoch=epoch, load_weightmap=FLAGS.load_weightmap, print_freq=FLAGS.print_freq)
        if epoch % 10 == 1: #so it validates at first epoch too
            optimizers = {'enc': opt_enc, 'dec': opt_dec, 'dis': opt_dis}
            losses = validate(val_loader, discr, vae, FLAGS.num_lat_dim, l1, optimizers, epoch=epoch, best_record=best_record,
                                val_imgs_sample_rate=FLAGS.val_imgs_sample_rate, val_save_to_img_file=True)
            tqdm.write("Epoch {} val losses: rec: {:.2E}, enc: {:.2E}, dec: {:.2E}, gan: {:.2E}".format(epoch, *losses))
        #sche_enc.step(val_loss_rec)
        #sche_dec.step(val_loss_rec)
        #sche_discr.step(val_loss_gan)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--full_glands', type=str2bool, default='y')

    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('-ndf', '--num_filt_discr', type=int, default=35, help='mcd number of filters for unet conv layers')
    parser.add_argument('-ngf', '--num_filt_gen', type=int, default=35, help='mcd number of filters for unet conv layers')
    parser.add_argument('-nz', '--num_lat_dim', type=int, default=4096, help='size of the latent z vector')

    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--learning_rate_patience', default=50, type=int)
    #parser.add_argument('--momentum', default=0.95, type=float) #for SGD not ADAM
    parser.add_argument('--class_weights', default=None)
    parser.add_argument('--load_weightmap', type=str2bool, default=False)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--val_batch_size', default=20, type=int)
    parser.add_argument('--val_imgs_sample_rate', default=0.05, type=float)
    parser.add_argument('--snapshot', default='')

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/rittscher/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')
    parser.add_argument('--checkpoint_folder', default='checkpoints', type=str, help='checkpoint folder')
    parser.add_argument('--resume', default='', type=str, help='which checkpoint file to resume the training')

    FLAGS, unparsed = parser.parse_known_args()
    if unparsed: warnings.warn("Unparsed arguments")
    main(FLAGS)
