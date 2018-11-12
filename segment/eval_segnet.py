#!/usr/bin/python

import datetime
import os, sys
import random
import argparse
from pathlib import Path
from numbers import Integral
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

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



import warnings

from models import *
from utils import on_cluster, get_time_stamp, check_mkdir, str2bool, \
    evaluate_multilabel, colorize, AverageMeter, get_flags
from prostate_dataset import SegDataset, AugDataset, TestDataset

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
    sys.path.append("/gpfs0/users/win/achatrian/cancer_phenotype")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype")
from clr import CyclicLR


def test(loader, net, criterion):

    test_loss = AverageMeter()
    test_preds = []
    test_labels = []

    net.eval()
    losses = []
    with no_grad():
        for i, data in enumerate(tqdm(loader)):
            inputs, labels = data
            # labels = labels.squeeze()
            N = inputs.size(0)
            inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
            labels = Variable(labels).cuda() if cuda.is_available() else Variable(labels)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.data.item())

            test_loss.update(loss.data.item(), N)

            # Compute metrics:
            test_preds.append(outputs.data.cpu().numpy())
            test_labels.append(labels.data.cpu().numpy())

    predictions = np.concatenate(test_preds, axis=0)
    gts = np.concatenate(test_labels, axis=0)
    acc, acc_cls, dice, dice_cls = evaluate_multilabel(predictions, gts)
    return test_loss.avg, acc, acc_cls, dice, dice_cls, losses

exp_name=''
def main(FLAGS):

    if torch.cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral):
            cuda.set_device(FLAGS.gpu_ids)
        else:
            cuda.set_device(FLAGS.gpu_ids[0])

    if FLAGS.chkpt_dir:
        ckpt_path = Path(FLAGS.chkpt_dir)  # scope? does this change inside train and validate functs as well?
        try:
            LOADEDFLAGS = get_flags(str(ckpt_path / ckpt_path.parent.name) + ".txt")
            FLAGS.num_filters = LOADEDFLAGS.num_filters
            FLAGS.num_class = LOADEDFLAGS.num_class
            FLAGS.image_size = LOADEDFLAGS.image_size
        except FileNotFoundError:
            pass

    print("Running on GPUs: {} - {} CPU workers".format(FLAGS.gpu_ids, FLAGS.workers))

    parallel = not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and cuda.is_available()
    inputs = {'num_classes' : FLAGS.num_class, 'num_channels' : FLAGS.num_filters}
    if FLAGS.network_id == "UNet1":
        net = UNet1(**inputs)
    elif FLAGS.network_id == "UNet2":
        net = UNet2(**inputs)
    elif FLAGS.network_id == "UNet3":
        net = UNet3(**inputs)
    elif FLAGS.network_id == "UNet4":
        net = UNet4(**inputs)
    if torch.cuda.is_available():
        net = net.cuda()
    if parallel:
        net = DataParallel(net, device_ids=FLAGS.gpu_ids).cuda()

    net.eval()

    FLAGS.batch_size = 1

    state_dict = load(str(Path(FLAGS.chkpt_dir) / exp_name / FLAGS.snapshot))
    if not parallel:
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    net.load_state_dict(state_dict)

    criterion = BCEWithLogitsLoss(size_average=True)

    bad_folds = \
        """17_A047-47800_15J+-+2017-05-11+08.12.44_TissueTrain_(1.00,43956,9763,5196,4388)_img_2070,2340.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_2488,1057.png
EU_26162_16_9x_HandE+-+2017-11-28+11.03.04_TissueTrain_(1.00,27962,65821,5548,5102)_img_3500,2850.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_345,235.png
EU_26162_16_9x_HandE+-+2017-11-28+11.03.04_TissueTrain_(1.00,27962,65821,5548,5102)_img_3500,3054.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_5486,838.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_1726,2290.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,44913,31266,5189,4018)_img_2049,2049.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_5906,861.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_2049,1.png
17_A047-47800_15J+-+2017-05-11+08.12.44_TissueTrain_(1.00,43956,9763,5196,4388)_img_1855,2340.png
17_A047-47800_15J+-+2017-05-11+08.12.44_TissueTrain_(1.00,43956,9763,5196,4388)_img_2524,2340.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_2296,2290.png
EU_35928_15_4N_HandE+-+2017-11-28+12.08.09_TissueTrain_(1.00,82097,43128,4845,4948)_img_2049,1.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_5906,1721.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_4097,4097.png
EU_38663_17_7G_HandE+-+2017-11-28+11.48.55_TissueTrain_(1.00,59274,36066,8452,4001)_img_2049,1.png
EU_35928_15_4N_HandE+-+2017-11-28+12.08.09_TissueTrain_(1.00,43070,65220,7175,6437)_img_6145,4097.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_4097,1.png
EU_37106_15_2K_HandE+-+2017-11-28+11.43.55_TissueTrain_(1.00,63062,11375,5880,10230)_img_1,1.png
EU_42619_15_2H_HandE+-+2017-11-28+11.31.12_TissueTrain_(1.00,35184,58159,4507,4194)_img_2459,462.png
17_A047-28956_156N+-+2017-05-11+09.57.53_TissueTrain_(1.00,35822,78102,5756,5827)_img_2347,3779.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_1202,2222.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_1695,2542.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_1,1.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_1112,1366.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_1,2049.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_2049,1.png
17_A047-8544_16J+-+2017-05-11+08.49.27_TissueTrain_(1.00,31805,24400,4623,5018)_img_2090,2970.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_2049,2049.png
17_A047-47800_15J+-+2017-05-11+08.12.44_TissueTrain_(1.00,43956,9763,5196,4388)_img_3148,1439.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,44913,31266,5189,4018)_img_768,595.png
17_A047-8544_16J+-+2017-05-11+08.49.27_TissueTrain_(1.00,31805,24400,4623,5018)_img_1965,599.png
EU_37747_14_5S_HandE+-+2017-11-28+13.04.58_TissueTrain_(1.00,71629,58933,4041,3717)_img_2049,2049.png
17_A047-28956_156N+-+2017-05-11+09.57.53_TissueTrain_(1.00,35822,78102,5756,5827)_img_3708,3779.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_665,1858.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_1,4097.png
EU_36244_16_9A_HandE+-+2017-11-28+12.41.47_TissueTrain_(1.00,55018,52345,8888,4200)_img_4097,1.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_3849,3236.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_2220,4290.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,44913,31266,5189,4018)_img_1340,149.png
17_A047-8544_16J+-+2017-05-11+08.49.27_TissueTrain_(1.00,31805,24400,4623,5018)_img_1388,2015.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_2488,2054.png
17_A047-47800_15J+-+2017-05-11+08.12.44_TissueTrain_(1.00,43956,9763,5196,4388)_img_2855,1450.png
17_A047-47800_15J+-+2017-05-11+08.12.44_TissueTrain_(1.00,43956,9763,5196,4388)_img_3148,1583.png
EU_36244_16_9A_HandE+-+2017-11-28+12.41.47_TissueTrain_(1.00,55018,52345,8888,4200)_img_4845,2152.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_2049,2049.png
17_A047-8544_16J+-+2017-05-11+08.49.27_TissueTrain_(1.00,31805,24400,4623,5018)_img_1107,2906.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_1295,2222.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_6145,1.png
17_A047-19575_162E+-+2017-05-11+08.34.36_TissueTrain_(1.00,31102,45887,5447,4555)_img_167,2507.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_2483,602.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_1160,3716.png
EU_36244_16_9A_HandE+-+2017-11-28+12.41.47_TissueTrain_(1.00,55018,52345,8888,4200)_img_3891,1177.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,44913,31266,5189,4018)_img_4097,2049.png
17_A047-19575_162E+-+2017-05-11+08.34.36_TissueTrain_(1.00,31102,45887,5447,4555)_img_1,2049.png
17_A047-28956_156N+-+2017-05-11+09.57.53_TissueTrain_(1.00,35822,78102,5756,5827)_img_3056,3779.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_3022,1006.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_464,2222.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_6145,2049.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_2408,2290.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_312,2120.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_1596,2290.png
EU_36244_16_9A_HandE+-+2017-11-28+12.41.47_TissueTrain_(1.00,55018,52345,8888,4200)_img_4172,1738.png
17_A047-28956_156N+-+2017-05-11+09.57.53_TissueTrain_(1.00,35822,78102,5756,5827)_img_4097,1.png
17_A047-10719_16L+-+2017-05-11+08.56.52_TissueTrain_(1.00,55114,25216,4536,4338)_img_678,1118.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_1,2049.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_2245,4303.png
17_A047-28956_156N+-+2017-05-11+09.57.53_TissueTrain_(1.00,35822,78102,5756,5827)_img_3676,196.png
EU_29542_16_1S_HandE+-+2017-11-28+12.49.46_TissueTrain_(1.00,75659,33158,4703,4374)_img_2049,2049.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_1208,4303.png
17_A047-28956_156N+-+2017-05-11+09.57.53_TissueTrain_(1.00,35822,78102,5756,5827)_img_930,1042.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,66060,20075,6553,4555)_img_4097,1.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_1,2049.png
EU_38663_17_7G_HandE+-+2017-11-28+11.48.55_TissueTrain_(1.00,59274,36066,8452,4001)_img_1607,8.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,44913,31266,5189,4018)_img_1,2049.png
EU_37747_14_5S_HandE+-+2017-11-28+13.04.58_TissueTrain_(1.00,71629,58933,4041,3717)_img_1993,1447.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,43743,25832,5070,4270)_img_3022,2222.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_2049,4097.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,44913,31266,5189,4018)_img_1,1.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,66060,20075,6553,4555)_img_1674,404.png
EU_42619_15_2H_HandE+-+2017-11-28+11.31.12_TissueTrain_(1.00,35184,58159,4507,4194)_img_2459,1370.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,44913,31266,5189,4018)_img_2049,1.png
EU_38663_17_7G_HandE+-+2017-11-28+11.48.55_TissueTrain_(1.00,38689,23348,4258,4405)_img_2049,1.png
EU_37747_14_5S_HandE+-+2017-11-28+13.04.58_TissueTrain_(1.00,71629,58933,4041,3717)_img_1993,1669.png
EU_38663_17_7G_HandE+-+2017-11-28+11.48.55_TissueTrain_(1.00,38689,23348,4258,4405)_img_1044,283.png
EU_38663_17_7G_HandE+-+2017-11-28+11.48.55_TissueTrain_(1.00,38689,23348,4258,4405)_img_2210,11.png
17_A047-47800_15J+-+2017-05-11+08.12.44_TissueTrain_(1.00,43956,9763,5196,4388)_img_2049,2049.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_615,3236.png
17_A047-8544_16J+-+2017-05-11+08.49.27_TissueTrain_(1.00,31805,24400,4623,5018)_img_2575,1843.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_1413,1289.png
EU_35928_15_4N_HandE+-+2017-11-28+12.08.09_TissueTrain_(1.00,43070,65220,7175,6437)_img_4097,2049.png
EU_18387_14_1E_HandE_TissueTrain_(1.00,66060,20075,6553,4555)_img_2049,2049.png
EU_7189_16_2D_HandE_TissueTrain_(1.00,56725,8687,6565,5284)_img_4517,3236.png
EU_29542_16_1S_HandE+-+2017-11-28+12.49.46_TissueTrain_(1.00,75659,33158,4703,4374)_img_2151,2326.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_5906,2319.png
EU_52005_15_2I_HandE+-+2017-11-28+11.23.02_TissueTrain_(1.00,82176,39556,6644,4967)_img_4551,1035.png
EU_30479_16_G_HandE+-+2017-11-28+12.31.57_TissueTrain_(1.00,49869,46001,7954,6351)_img_5906,1877.png
EU_52005_15_2I_HandE+-+2017-11-28+11.23.02_TissueTrain_(1.00,82176,39556,6644,4967)_img_4596,523.png
17_A047-4519_1614P+-+2017-05-11+09.50.49_TissueTrain_(1.00,32465,16322,5478,4691)_img_2250,2643.png """
    bad_folds = bad_folds.replace(" ", "")
    bad_folds = bad_folds.split('\n')

    dataset = TestDataset(FLAGS.data_dir, tile_size=FLAGS.image_size, bad_folds=bad_folds)
    loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers)

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Memory: Image size: {}, Batch size: {},  Net filter num: {}".format(FLAGS.image_size, FLAGS.batch_size,
                                                                               FLAGS.num_filters))
    print("Network has {} parameters".format(params))
    print("Testing ...")

    test_loss, acc, acc_cls, dice, dice_cls, losses = test(loader, net, criterion)

    test_msg = 'Test result - test loss: {:.5f}, acc: {:.5f}, acc_cls: {}, dice {:.5f}, dice_cls: {}\n'.format(
        test_loss, acc, acc_cls, dice, dice_cls)
    print(test_msg)

    loss_idx = list(np.argsort(losses))
    N = 100
    worst_imgs = [os.path.basename(dataset.file_list[i]) for i in loss_idx[-N:]]
    best_imgs = [os.path.basename(dataset.file_list[i]) for i in loss_idx[0:N]]

    with open(FLAGS.chkpt_dir + "/test_result_" + FLAGS.snapshot[:--4] + ".txt", 'w') as save_file:
        save_file.write(test_msg)
        save_file.write(" -------- Worst images: --------\n")
        for name in worst_imgs:
            save_file.write(name + '\n')
        save_file.write(" -------- Best images: ---------\n")
        for name in best_imgs:
            save_file.write(name + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--downsample', type=float, default=2.0)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--augment', type=int, default=0)

    parser.add_argument('--network_id', type=str, default="UNet4")
    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)

    parser.add_argument('--val_imgs_sample_rate', default=0.05, type=float)
    parser.add_argument('--snapshot', default='')

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')
    parser.add_argument('--chkpt_dir', default='', type=str, help='checkpoint folder')
    "/gpfs0/well/win/users/achatrian/cancer_phenotype/Dataset/pix2pix/generated"

    FLAGS = parser.parse_args()
    main(FLAGS)