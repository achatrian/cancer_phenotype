import os, sys
import argparse
from warnings import warn
from pathlib import Path
from numbers import Integral

import numpy as np
from scipy.stats import kurtosis, skew, mode, entropy
import torch
from torch import cuda, load
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import cv2
from tqdm import tqdm
from imageio import imwrite


from torchvision.models import inception_v3

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

from phenotyping.feature_utils import FeatureExtractorInception, summary_net_features_torch, \
    make_thumbnail_fullcut, create_sprite_image, to_tensor, augment_glands
from ck5.inception import Inception
from phenotyping.gland_dataset import GlandDataset
from segment.utils import check_mkdir

exp_name = ''


def main(FLAGS):
    # Set segmentation network up
    if cuda.is_available():
        if isinstance(FLAGS.gpu_ids, Integral) and FLAGS.gpu_ids >= 0:
            # for queues, let the automatic allocator choose your gpu
            cuda.set_device(FLAGS.gpu_ids)
        else:
            cuda.set_device(FLAGS.gpu_ids[0])

    FLAGS.save_dir = os.path.expanduser(FLAGS.save_dir)
    check_mkdir(FLAGS.save_dir)

    ckpt_path = Path(FLAGS.ck_model).parent  # scope? does this change inside train and validate functs as well?
    parallel = not isinstance(FLAGS.gpu_ids, Integral) and len(FLAGS.gpu_ids) > 1 and cuda.is_available()

    # Model:
    net = inception_v3(pretrained=False, num_classes=FLAGS.num_class, aux_logits=True, transform_input=True)
    if cuda.is_available():
        net = net.cuda()
    print("Loaded inception model")

    state_dict = load(os.path.expanduser(FLAGS.ck_model))
    state_dict = {key[11:]: value for key, value in
                  state_dict.items()}  # never loading parallel as feature_net is later loaded as parallel
    net.load_state_dict(state_dict)
    net.eval()

    features_net = FeatureExtractorInception(net)
    if cuda.is_available():
        features_net = features_net.cuda()
    if parallel:
        features_net = DataParallel(features_net, device_ids=FLAGS.gpu_ids).cuda()
    features_net.eval()  # extracting features only

    datasets = [GlandDataset(FLAGS.data_dir, mode, 299) for mode in ['train', 'val', 'test']]
    loaders = [DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.workers) for ds in datasets]

    tqdm.write("Generating features and thumbnails ...")
    X = []  # feature matrix
    #X_alg = []
    thumbnails = []
    th_size = (FLAGS.thumbnail_size,) * 2
    labels = []
    paths = []

    N = FLAGS.augment_num
    for loader, mode in zip(loaders, ['train', 'val', 'test']):
        tqdm.write("... for {} data ...".format(mode))
        # Use train test and validate datasets
        for i, data in enumerate(tqdm(loader)):

            inputs = data[0]
            gts = data[1]
            colours_n_size = data[2].numpy()
            label_batch = data[3]
            label_batch = list(label_batch.numpy())
            M = inputs.shape[0]  # remember how many different gland images are there

            paths.extend(data[4])

            assert np.any(np.array(inputs.shape))  # non empty data

            imgs = inputs.numpy().transpose(0, 2, 3, 1)
            gts = gts.numpy().transpose(0, 2, 3, 1)
            imgs, gts = augment_glands(imgs, gts, N=N)  # consecutive images are augmented versions of first image

            with torch.no_grad():  # speeds up computations and uses less RAM
                inputs = to_tensor(imgs)
                inputs = Variable(inputs)
                inputs = inputs.cuda() if cuda.is_available() else inputs
                feature_maps = features_net(inputs)  # Channel as last dimension

                assert feature_maps  # non empty feature map list
                gts = to_tensor(gts)
                gts = gts.cuda() if cuda.is_available() else gts
                glands_features = summary_net_features_torch(gts, feature_maps[0:-1])
                fc = feature_maps[-1]

                if N:
                    # started with M images, added M*N with augmentation, now take mean of N+1 responses
                    glands_features = glands_features.reshape((glands_features.shape[0] // (N + 1),
                                                               (N + 1), -1))
                    fc = fc.reshape((fc.shape[0] // (N + 1), (N + 1), -1))
                    glands_features = glands_features.mean(1)
                    fc = fc.mean(1)

                # Append final feats:
                glands_features = torch.cat((glands_features, fc), dim=1)  # add fc layer at end of network

            assert glands_features.shape[0] == M  # ensure there is one feature vector per image

            bad_imgs = []
            #for j, (gl_features, gl_alg_features) in enumerate(zip(glands_features, alg_feats)):
            for j, gl_features in enumerate(glands_features):
                # Checks FIXME should not have to do this
                gl_features = gl_features.cpu().numpy()
                if gl_features.dtype != np.float32 or gl_features.ndim != 1:
                    bad_imgs.append(j)  # nan response
                    continue
                X.append(gl_features)  # feature vector
                #X_alg.append(gl_alg_features)
                X[-1] = np.concatenate((gl_features, colours_n_size[j, ...]))

            # Save original images
            gts = gts[0::(N + 1), ...].cpu().numpy().squeeze()
            for j, (img, gt, label) in enumerate(zip(imgs[0::(N + 1), ...], gts, label_batch)):
                # only original images
                if j in bad_imgs:
                    del label_batch[j]
                    continue  # skip if nan response
                gt = gt.squeeze()
                tn = make_thumbnail_fullcut(img, gt, th_size, label=label)  # for full gland images
                thumbnails.append(tn)

                if i == 0:
                    imwrite(str(Path(FLAGS.save_dir).expanduser() / "thumbnail_test.png"), tn)

            labels.extend(label_batch)

            assert len(X) == len(thumbnails)
            assert len(X) == len(labels)

    # Save features
    model_name = str(Path(FLAGS.ck_model).name)[:-4]
    X = np.array(X)
    header = "mean_1,std_1,max_1,mean_2,std_2,max_2,..._{}x{}_{}".format(X.shape[0], X.shape[1], model_name)
    save_file = Path(FLAGS.save_dir).expanduser() / ("feats_" + model_name + ".csv")
    with open(str(save_file), 'w') as feature_file:
        np.savetxt(feature_file, X, delimiter=' ', header=header)
    print("Saved feature matrix ({}x{})".format(*X.shape))

    # Save thumbnails
    thumbnails = np.array(thumbnails)
    spriteimage = create_sprite_image(thumbnails).astype(np.uint8)
    save_file = Path(FLAGS.save_dir).expanduser() / ("sprite_" + model_name + ".png")
    imwrite(save_file, spriteimage)
    save_file = Path(FLAGS.save_dir).expanduser() / (
            "thumbnails_" + model_name + ".npy")  # format for saving numpy data (not compressed)
    # Revert to RGB:
    thumbnails = thumbnails.astype(np.uint8)
    with open(str(save_file), 'wb') as thumbnails_file:  # it uses pickle, hence need to open file in binary mode
        np.save(thumbnails_file, thumbnails)
    print("Saved {} ({}x{} thumbnails".format(*thumbnails.shape))

    labels = np.array(labels)
    # Save labels:
    save_file = Path(FLAGS.save_dir).expanduser() / ("labels_" + model_name + ".tsv")
    with open(str(save_file), 'w') as labels_file:
        np.savetxt(labels_file, labels, delimiter='\t')
    print(
        "Saved labels ({}) - {} tumour and {} gland".format(labels.size, np.sum(labels == 2), np.sum(labels == 3)))
    print("Done!")

    # Save paths
    save_file = Path(FLAGS.save_dir).expanduser() / ("paths_" + model_name + ".csv")
    with open(str(save_file), 'w') as paths_file:
        for item in paths:
            paths_file.write("{}\n".format(item))

    print("Saved paths ({})".format(len(paths)))


def thumbnails_size_check(str_size):
    size = int(str_size)
    if size > 100:
        warn("Max allowed thumbnail size is 100")
        size = 100
    return size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-ckm', '--ck_model', required=True, type=str, help='ck model file')
    parser.add_argument('-sd', '--save_dir', default="/gpfs0/well/win/users/achatrian/cancer_phenotype/Results")
    parser.add_argument('--thumbnail_size', default=64, type=thumbnails_size_check)
    parser.add_argument('--augment_num', type=int, default=0)

    parser.add_argument('--network_id', type=str, default="UNet4")
    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/win/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--gpu_ids', default=-1, nargs='+', type=int, help='gpu ids')
    parser.add_argument('--workers', default=2, type=int, help='the number of workers to load the data')

    parser.add_argument('--num_class', type=int, default=3)
    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)