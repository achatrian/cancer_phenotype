#!/usr/bin/python

#visualise images
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import from_numpy, FloatTensor
from utils import on_cluster, colorize, evaluate_multilabel, colorize, dice_loss, MultiLabelSoftDiceLoss
import time
import imageio


from prostate_dataset import ProstateDataset


def test_dataset(dataset,visualize, viewnum=50, testgt=False, testshape=False):
    print("Number of tiles = {}".format(len(dataset.img_files)))
    print("Dataset length = {}".format(len(dataset)))

    for idx, (img, gt) in enumerate(dataset):

        shape = (512, 512)
        if testshape:
            if img.shape[1:] != shape or gt.shape[1:] != shape:
                print("Wrong dims: i{}, dims ({}, {})".format(idx, *img.shape[1:]))
            if img.shape != gt.shape:
                print("img and gt dims don't match: i{}".format(idx))
        if not np.any(gt > 0) and testgt:
            print(dataset.gt_files[idx])
            imageio.imwrite("~/Desktop/CheckGoodGTs/{}.png".format(idx), img.numpy().transpose(1,2,0).astype(np.uint8))
            imageio.imwrite("~/Desktop/CheckGoodGTs/{}_gt.png".format(idx), colorize(gt.numpy()))
            if visualize:
                gt = gt.numpy().transpose(1,2,0)
                img = img.numpy().transpose(1,2,0)
                fig, axes = plt.subplots(1,2)
                axes[0].imshow(img[:,:,0])
                axes[1].imshow(gt[:,:,0])
                plt.show()


def test_evaluate(preds, gts):
    acc, acc_cls, dice, dice_cls = evaluate_multilabel(preds, gts)
    print("Class mean accuracy = {}".format(acc))
    print("Class accuracies = {}".format(acc_cls))
    print("Class mean dice = {}".format(dice))
    print("Class dice = {}".format(dice_cls))

def test_loss(preds, gts):
    if preds.shape[1] < 2:
        out = dice_loss(1/(1+np.exp(-preds)), gts, from_numpy(np.array([1 for i in range(preds.shape[1])])).type(FloatTensor))
    else:
        out = dice_loss(preds, gts, from_numpy(np.array([1, 1, 1])).type(FloatTensor))
    print("Loss = {}".format(out))
    criterion = MultiLabelSoftDiceLoss(num_class=preds.shape[1])
    out = criterion(preds, gts)
    print("Lossclass = {}".format(out))


visualize=False
if on_cluster():
    dir = "/gpfs0/well/win/users/achatrian/ProstateCancer/Dataset"
else:
    dir = "/Volumes/A.CH.EXDISK1/Projects/Dataset"
dataset = ProstateDataset(dir, 'train',  num_class=1, grayscale=True, down=4.0, out_size=512)

test_dataset(dataset, visualize, testshape=True)

train_loader = DataLoader(dataset)

imgs, gts = next(iter(train_loader))
print(imgs.shape, gts.shape)

gts_np = gts.numpy()
test_evaluate(gts_np, gts_np)

test_loss(gts, gts)
