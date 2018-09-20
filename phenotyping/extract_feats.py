
import os, sys
import random
import argparse
import numpy as np
from pathlib import Path
from numbers import Integral
from torch import cuda, load
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import cv2

from gland_dataset import GlandPatchDataset

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
    sys.path.append(os.path.expanduser('~') + 'ProstateCancer')
else:
    sys.path.append(os.path.expanduser('~') + '/Documents/Repositories/ProstateCancer')

from mymodel.utils import on_cluster, get_time_stamp, check_mkdir, str2bool, \
    evaluate_multilabel, colorize, AverageMeter, get_flags
from mymodel.models import UNet1, UNet2, UNet3
exp_name = ''


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if not self.extracted_layers:
                continue
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
                self.extracted_layers.remove(name)
        return outputs


def get_gland_bb(gt):
    gt2, contours, hierarchy = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contour) for contour in contours]
    gland_contour = contours[areas.index(max(areas))]
    x, y, w, h = cv2.boundingRect(gland_contour)
    return x, y, w, h


def main(FLAGS):

    #Set up network
    if cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral):
            cuda.set_device(FLAGS.gpu_ids)
        else:
            cuda.set_device(FLAGS.gpu_ids[0])

    if FLAGS.checkpoint_folder:
        ckpt_path = Path(FLAGS.checkpoint_folder)  # scope? does this change inside train and validate functs as well?
        try:
            LOADEDFLAGS = get_flags(str(ckpt_path / ckpt_path.parent.name) + ".txt")
            FLAGS.network_id = LOADEDFLAGS.network_id
            FLAGS.num_filters = LOADEDFLAGS.num_filters
            FLAGS.num_class = LOADEDFLAGS.num_class
        except FileNotFoundError:
            print("Settings could not be loaded")

    parallel = not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and cuda.is_available()
    inputs = {'num_classes' : FLAGS.num_class, 'num_channels' : FLAGS.num_filters}
    if FLAGS.network_id == "UNet1":
        net = UNet1(**inputs).cuda() if cuda.is_available() else UNet1(**inputs)
    elif FLAGS.network_id == "UNet2":
        net = UNet2(**inputs).cuda() if cuda.is_available() else UNet2(**inputs)
    elif FLAGS.network_id == "UNet3":
        net = UNet3(**inputs).cuda() if cuda.is_available() else UNet3(**inputs)

    dev = None if cuda.is_available() else 'cpu'
    state_dict = load(str(ckpt_path / exp_name / FLAGS.snapshot), map_location=dev)
    if not parallel:
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    net.load_state_dict(state_dict)

    feature_blocks = ['enc4', 'enc5', 'enc6', 'center']
    downsamplings = [4, 5, 6, 6]  # how many times the original dimensions have been downsampled at output of module

    features_net = FeatureExtractor(net, feature_blocks)
    if parallel:
        features_net = DataParallel(features_net, device_ids=FLAGS.gpu_ids).cuda()
    features_net.eval()  # extracting features only

    dataset = GlandPatchDataset(FLAGS.data_dir, "train")
    loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.workers)


    increase_bb = 0.1
    for i, data in enumerate(loader):
        inputs = data[0]
        gt = data[1]

        inputs = Variable(inputs).cuda() if cuda.is_available() else Variable(inputs)
        feature_maps = features_net(inputs) #Channel as last dimension

        #Get features only around gland
        pos = np.array(get_gland_bb(gt))
        pos[0] = max(0, pos[0] - pos[0] * increase_bb)  # x
        pos[1] = max(0, pos[1] - pos[1] * increase_bb)  # y
        pos[2] = max(0, pos[2] + pos[2] * increase_bb)  # w
        pos[3] = max(0, pos[3] + pos[3] * increase_bb)  # h
        pos_gland = {block: pos / dwnsmpl for block, dwnsmpl in zip(feature_blocks, downsamplings)}
        gland_features = []
        for fm, pos in zip(feature_maps, pos_gland.values()):
            if fm.shape[1] > 10:
                x, y, w, h = pos
                fm = fm[y:y+h, x:x+w]
            gland_features.append(fm)













if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-chk', '--checkpoint_folder', required=True, type=str, help='checkpoint folder')
    parser.add_argument('-snp', '--snapshot', required=True, type=str, help='model file')

    parser.add_argument('--network_id', type=str, default="UNet1")
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/ProstateCancer/Dataset")
    parser.add_argument('--gpu_ids', default=0, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)



