#Get bad examples:

import os, sys
import argparse
import time
from tqdm import tqdm
from pathlib import Path
from numbers import Integral

import numpy as np
from torch import load, stack, cuda, no_grad
from torch.autograd import Variable
from torch.nn import DataParallel, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from imageio import imwrite, imread
from cv2 import cvtColor, COLOR_GRAY2RGB
import re

from prostate_dataset import ProstateDataset
from utils import on_cluster, check_mkdir, evaluate_multilabel, colorize, get_flags, str2bool
from utils import str2bool, MultiLabelSoftDiceLoss
from models import *

if on_cluster():
    sys.path.append("/gpfs0/users/win/achatrian/cancer_phenotype/korsuks_code")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype/korsuks_code")
from dataset_FusionNet import SegDataset

def save_examples(net, loader, criterion, savedir, save_images=True, save_weightmap=False, batch_size=None):
    dataset_metrics = {'loss': [], 'acc': [], 'acc_cls' : [], 'dice' : [], 'dice_cls' : []}
    saved_num = 0
    if save_weightmap:
        maplist_path = os.path.join(loader.dataset.dir, "{}_weightmaps.txt".format(savedir.split('/')[-1]))
        maplist_file = open(maplist_path, 'wt')
        print("Saving weight maps ...")
    for i, (inputs, gts) in enumerate(tqdm(loader)): #pass over whole validation dataset
        with no_grad(): #don't track variable history for backprop (to avoid out of memory)
            #NB @pytorch variables and tensors will be merged in the future
            inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
            gts = Variable(gts).cuda() if cuda.is_available() else Variable(gts)
            outputs = net(inputs)

        for j, (img, gt, pred) in enumerate(zip(inputs, gts, outputs)):
            #Save metrics for img
            loss = criterion(pred.unsqueeze(0), gt.unsqueeze(0)).data.item()
            img, gt, pred = img.cpu().numpy().astype(np.uint8), gt.cpu().numpy(), pred.cpu().numpy()  #convert to numpy
            img_metrics = list(evaluate_multilabel(pred[np.newaxis,...], gt[np.newaxis,...]))
            acc, acc_cls, dice, dice_cls = img_metrics
            for key, val in zip(dataset_metrics, [loss] + img_metrics):
                dataset_metrics[key].append(val)  #append values

            #Save img if network performs poorly (less than median)
            gt_rgb, pred_rgb = colorize(gt), colorize(pred)

            poor_performance = loss < np.median(dataset_metrics['loss']) or acc < np.median(dataset_metrics['acc']) or \
                        dice < np.median(dataset_metrics['dice'])
            if poor_performance and save_images:
                input_rgb = img.transpose(1,2,0) if img.shape[0] == 3 else cvtColor(img.transpose(1,2,0) , COLOR_GRAY2RGB)
                imwrite(savedir + "/{}{}_l{}_acc{}_dice{}.png".format(i, j , loss, acc, dice), input_rgb)
                imwrite(savedir + "/{}{}_l{}_acc{}_dice{}_gt.png".format(i, j , loss, acc, dice), gt_rgb)
                imwrite(savedir + "/{}{}_l{}_acc{}_dice{}_pred.png".format(i, j , loss, acc, dice), pred_rgb)
                saved_num += 1

            #Saving weightmap defined by union of gt and pred - intersection of gt and pred
            if save_weightmap:
                if not batch_size: raise ValueError("Need to provide batch size if want to save weight map")
                try:
                    idx = i*batch_size + j
                    gtpath = Path(loader.dataset.gt_files[idx])  #if dataloader was passed
                except AttributeError:
                    idx = i
                    gtpath = Path(loader.gt_files[idx])  #if dataset was passed

                #Compute union - intersection
                union = np.logical_or(gt_rgb, pred_rgb).astype(np.uint8)
                intersection = np.logical_and(gt_rgb, pred_rgb).astype(np.uint8)
                umi = union - intersection

                srch = re.search('_mask_([0-9\(\)]+),([0-9\(\)]+).png', str(gtpath))
                nums = (srch.group(1), srch.group(2))
                check_mkdir(str(gtpath.parents[1]/"weightmaps"))
                savepath = str(gtpath.parents[1]/"weightmaps"/"weightmap_{},{}.png".format(*nums))
                if os.path.exists(savepath):
                    continue    #don't remake if map already exists
                imwrite(savepath, umi*255) #0 or 255 (to be visible in preview as image)
                print(savepath, file = maplist_file)
    if save_weightmap:
        maplist_file.close()

    return dataset_metrics, saved_num


def main(FLAGS):
    if cuda.is_available():
        cuda.set_device(FLAGS.gpu_ids[0] if not isinstance(FLAGS.gpu_ids, Integral) else FLAGS.gpu_ids)

    p = Path(FLAGS.model_filepath)
    flags_filepath = str(p.parents[0]/p.parts[-3]) + '.txt'
    try:
        TRAINFLAGS = get_flags(flags_filepath)  #get training arguments for loading net etc.
        FLAGS.network_id = TRAINFLAGS.network_id
        FLAGS.num_class = TRAINFLAGS.num_class
        FLAGS.num_filters = TRAINFLAGS.num_filters
    except FileNotFoundError:
        inps = {'num_classes': FLAGS.num_class, 'num_channels': FLAGS.num_filters}

    #Setup
    inps = {'num_classes': FLAGS.num_class, 'num_channels': FLAGS.num_filters}
    if FLAGS.network_id == "UNet1":
        net = UNet1(**inps).cuda() if cuda.is_available() else UNet1(**inps) #possible classes are stroma, gland, lumen
    elif FLAGS.network_id == "UNet2":
        net = UNet2(**inps).cuda() if cuda.is_available() else UNet2(**inps)
    elif FLAGS.network_id == "UNet3":
        net = UNet3(**inps).cuda() if cuda.is_available() else UNet3(**inps) #possible classes are stroma, gland, lumen
    elif FLAGS.network_id == "UNet4":
        net = UNet4(**inps).cuda() if cuda.is_available() else UNet4(**inps)

    netdict = load(FLAGS.model_filepath, map_location = None if cuda.is_available() else'cpu')

    if not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and cuda.is_available():
        net = DataParallel(net, device_ids=FLAGS.gpu_ids).cuda()
    else:
        #NB there is a .module at beginning for DataParallel nets that is quite annyoing and needs to be removed ...
        netdict = {entry[7:] : tensor for entry, tensor in netdict.items()}

    net.load_state_dict(netdict)
    if cuda.is_available(): net = net.cuda()
    net.eval()

    #Loss
    if FLAGS.num_class>1:
        criterion = MultiLabelSoftMarginLoss(size_average=True) #loss
    else:
        criterion = BCEWithLogitsLoss(size_average=True)


    if cuda.is_available(): criterion = criterion.cuda()

    train_dataset = SegDataset(FLAGS.data_dir, "train")
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers)
    val_dataset = SegDataset(FLAGS.data_dir, "val")
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers)
    #test_dataset = ProstateDataset(FLAGS.data_dir, "test", TRAINFLAGS.image_size, down=TRAINFLAGS.downsample, grayscale=TRAINFLAGS.grayscale)
    #test_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers)

    #Check performance on train dataset and save poor performance images
    savedir = '/'.join(FLAGS.model_filepath.split('/')[:-2] + ["deploy_on_dataset"])
    check_mkdir(savedir)
    check_mkdir(savedir + "/train")
    check_mkdir(savedir + "/validate")
    check_mkdir(savedir + "/test")
    msg = "Saved {:d} {} images; dataset median loss: {}, mean acc: {}, mean dice: {}"

    print("Running on GPUs: {}".format(FLAGS.gpu_ids))
    print("Evaluating on training images ...")
    train_metrics, train_saved_num = save_examples(net, train_loader, criterion, savedir + "/train", save_images=FLAGS.save_images,
                                                    save_weightmap=FLAGS.save_weightmap, batch_size=FLAGS.batch_size)
    print(msg.format(train_saved_num, "train", np.median(train_metrics['loss']), np.median(train_metrics['acc']), np.median(train_metrics['dice'])))

    print("Evaluating on validation images ...")
    val_metrics, val_saved_num = save_examples(net, val_loader, criterion, savedir + "/validate", save_images=FLAGS.save_images,
                                               save_weightmap=FLAGS.save_weightmap, batch_size=FLAGS.batch_size)
    print(msg.format(val_saved_num, "val", np.median(val_metrics['loss']), np.median(val_metrics['acc']), np.median(val_metrics['dice'])))

    # print("Evaluating on test images ...")
    # test_metrics, test_saved_num = save_examples(net, test_loader, criterion, savedir + "/test", save_images=FLAGS.save_images,
    #                                              save_weightmap=FLAGS.save_weightmap, batch_size=FLAGS.batch_size)
    #
    # print(msg.format(test_saved_num, "test", np.median(test_metrics['loss']), np.median(test_metrics['acc']), np.median(test_metrics['dice'])))

    print("Done !")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-mf', '--model_filepath', type=str, required=True)
    parser.add_argument('-dd', '--data_dir', type=str, default="/Volumes/A-CH-EXDISK/Projects/Dataset")
    parser.add_argument('-si', '--save_images', type=str2bool, default=True)
    parser.add_argument('-sw', '--save_weightmap', type=str2bool, default=False)
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')


    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('-nf', '--num_filters', type=int, default=36, help='mcd number of filters for unet conv layers')
    parser.add_argument('--network_id', type=str, default="UNet4")

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
