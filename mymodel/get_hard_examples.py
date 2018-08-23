#Get bad examples:

import os
import argparse
import time
from pathlib import Path

import numpy as np
from torch import load, stack, cuda, no_grad
from torch.autograd import Variable
from torch.nn import DataParallel, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from imageio import imwrite, imread
from cv2 import cvtColor, COLOR_GRAY2RGB
import re

from prostate_dataset import ProstateDataset
from utils import check_mkdir, evaluate_multilabel, colorize, get_flags, str2bool
from utils import str2bool, MultiLabelSoftDiceLoss
from models import *

def save_examples(net, loader, criterion, savedir, save_images=True, save_weightmap=False, batch_size=None):
    dataset_metrics = {'loss': [], 'acc': [], 'acc_cls' : [], 'dice' : [], 'dice_cls' : []}
    saved_num = 0
    if save_weightmap:
        maplist_path = os.path.join(loader.dataset.dir, "{}_weightmaps.txt".format(savedir.split('/')[-1]))
        maplist_file = open(maplist_path, 'wt')
    for i, (inputs, gts) in enumerate(loader): #pass over whole validation dataset
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
                imwrite(savepath, umi*255) #0 or 255 (to be visible in preview as image)
                print(savepath, file=maplist_file)
        if save_weightmap:
            maplist_file.close()

    return dataset_metrics, saved_num, weightmap_files


def main(FLAGS):

    p = Path(FLAGS.model_filepath)
    flags_filepath = str(p.parents[0]/p.parts[-3]) + '.txt'
    TRAINFLAGS = get_flags(flags_filepath)  #get training arguments for loading net etc.
    inputs = {'num_classes' : TRAINFLAGS.num_class, 'num_channels' : TRAINFLAGS.num_filters,
                'grayscale' : TRAINFLAGS.grayscale, 'batchnorm' : TRAINFLAGS.batchnorm}
    #Setup
    if TRAINFLAGS.network_id == "UNet1":
        net = UNet1(**inputs).cuda() if cuda.is_available() else UNet1(**inputs) #possible classes are stroma, gland, lumen
    elif TRAINFLAGS.network_id == "UNet2":
        net = UNet2(**inputs).cuda() if cuda.is_available() else UNet2(**inputs)
    netdict = load(FLAGS.model_filepath, map_location = None if cuda.is_available() else'cpu')
    netdict = {entry[7:] : tensor for entry, tensor in netdict.items()} #NB there is a .module at beginning that is quite annyoing and needs to be removed ...
    net.load_state_dict(netdict)
    net.eval()

    #Loss
    if TRAINFLAGS.num_class>1:
        if TRAINFLAGS.losstype == 'ce':
            criterion = MultiLabelSoftMarginLoss(size_average=True, weight=TRAINFLAGS.class_weights) #loss
        else:
            criterion = MultiLabelSoftDiceLoss(num_class=TRAINFLAGS.num_class, weights=TRAINFLAGS.class_weights)
    else:
        if TRAINFLAGS.losstype == 'ce':
            criterion = BCEWithLogitsLoss(size_average=True, weight=TRAINFLAGS.class_weights)
        else:
            criterion = MultiLabelSoftDiceLoss(num_class=TRAINFLAGS.num_class, weights=TRAINFLAGS.class_weights)

    if cuda.is_available(): criterion = criterion.cuda()

    train_dataset = ProstateDataset(FLAGS.data_dir, "train", TRAINFLAGS.image_size, down=TRAINFLAGS.downsample, grayscale=TRAINFLAGS.grayscale, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers)
    val_dataset = ProstateDataset(FLAGS.data_dir, "validate", TRAINFLAGS.image_size, down=TRAINFLAGS.downsample, grayscale=TRAINFLAGS.grayscale)
    val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers)
    test_dataset = ProstateDataset(FLAGS.data_dir, "test", TRAINFLAGS.image_size, down=TRAINFLAGS.downsample, grayscale=TRAINFLAGS.grayscale)
    test_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers)

    #Check performance on train dataset and save poor performance images
    savedir = '/'.join(FLAGS.model_filepath.split('/')[:-2] + ["deploy_on_dataset"])
    check_mkdir(savedir)
    check_mkdir(savedir + "/train")
    check_mkdir(savedir + "/validate")
    check_mkdir(savedir + "/test")
    msg = "Saved {:d} {} images; dataset median loss: {}, mean acc: {}, mean dice: {}"

    print("Evaluating on training images ...")
    train_metrics, train_saved_num = save_examples(net, train_loader, criterion, savedir + "/train", save_images=FLAGS.save_images,
                                                    save_weightmap=FLAGS.save_weightmap, batch_size=FLAGS.batch_size)
    print(msg.format(train_saved_num, "train", np.median(train_metrics['loss']), np.median(train_metrics['acc']), np.median(train_metrics['dice'])))

    print("Evaluating on validation images ...")
    val_metrics, val_saved_num = save_examples(net, val_loader, criterion, savedir + "/validate", save_images=FLAGS.save_images,
                                               save_weightmap=FLAGS.save_weightmap, batch_size=FLAGS.batch_size)
    print(msg.format(val_saved_num, "val", np.median(val_metrics['loss']), np.median(val_metrics['acc']), np.median(val_metrics['dice'])))

    print("Evaluating on test images ...")
    test_metrics, test_saved_num = save_examples(net, test_loader, criterion, savedir + "/test", save_images=FLAGS.save_images,
                                                 save_weightmap=FLAGS.save_weightmap, batch_size=FLAGS.batch_size)

    print(msg.format(test_saved_num, "test", np.median(test_metrics['loss']), np.median(test_metrics['acc']), np.median(test_metrics['dice'])))

    print("Done !")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-mf', '--model_filepath', type=str, required=True)
    parser.add_argument('-dd', '--data_dir', type=str, default="/Volumes/A.CH.EXDISK1/Projects/Dataset")
    parser.add_argument('-si', '--save_images', type=str2bool, default=True)
    parser.add_argument('-sw', '--save_weightmap', type=str2bool, default=False)


    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')


    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
