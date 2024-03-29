import sys
import argparse
from warnings import warn
from pathlib import Path
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from imageio import imwrite


def on_cluster():
    import socket, re
    hostname = socket.gethostname()
    matches = []
    matches.append(re.search("jalapeno(\w\w)?.fmrib.ox.ac.uk", hostname))
    matches.append(re.search("cuda(\w\w)?.fmrib.ox.ac.uk", hostname))
    matches.append(re.search("login(\w\w)?.cluster", hostname))
    matches.append(re.search("gpu(\w\w)?", hostname))
    matches.append(re.search("compG(\w\w\w)?", hostname))
    matches.append(re.search("compE(\w\w\w)?", hostname))
    matches.append(re.search("compF(\w\w\w)?", hostname))
    matches.append(re.search("compA(\w\w\w)?", hostname))
    matches.append(re.search("compB(\w\w\w)?", hostname))
    matches.append(re.search("compC(\w\w\w)?", hostname))
    matches.append(re.search("rescomp(\w)?", hostname))
    return any(matches)

if on_cluster():
    sys.path.append("/users/rittscher/achatrian/cancer_phenotype")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype")

from segment.utils import check_mkdir, str2bool
from repo_old.phenotyping import feature_extraction, \
    make_thumbnail_fullcut, create_sprite_image, augment_glands
from repo_old.phenotyping import GlandDataset, GlandPatchDataset

exp_name = ''


def main(FLAGS):
    if FLAGS.full_glands:
        datasets = [GlandPatchDataset(FLAGS.data_dir, mode, 299) for mode in ['train', 'test']]
    else:
        datasets = [GlandDataset(FLAGS.data_dir, mode, 299) for mode in ['train', 'test']]
    loaders = [DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=False, drop_last=True, num_workers=FLAGS.workers) for ds in datasets]

    tqdm.write("Generating features and thumbnails ...")
    X = []  # feature matrix
    thumbnails = []
    th_size = (FLAGS.thumbnail_size,) * 2
    labels = []
    paths = []

    check_mkdir(FLAGS.save_dir)
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
            paths.extend(data[4])
            M = inputs.shape[0]  # remember how many different gland images are there

            assert np.any(np.array(inputs.shape))  # non empty data

            imgs = inputs.numpy().transpose(0, 2, 3, 1)
            gts = gts.numpy().transpose(0, 2, 3, 1).squeeze()
            imgs, gts = augment_glands(imgs, gts, N=N)  # consecutive images are augmented versions of first image

            handcraft_feats = feature_extraction(imgs, gts)

            handcraft_feats = handcraft_feats.reshape((handcraft_feats.shape[0] // (N + 1), (N + 1), -1))
            handcraft_feats = handcraft_feats.mean(axis=1)

            assert handcraft_feats.shape[0] == M  # ensure there is one feature vector per image

            bad_imgs = []
            #for j, (gl_features, gl_alg_features) in enumerate(zip(glands_features, alg_feats)):
            for j, gl_features in enumerate(handcraft_feats):
                # Checks FIXME should not have to do this
                X.append(gl_features)  # feature vector
                X[-1] = np.concatenate((gl_features, colours_n_size[j, ...]))

            # Save original images
            gts = gts[0::(N + 1), ...]
            for j, (img, gt, label) in enumerate(zip(imgs[0::(N + 1), ...], gts, label_batch)):
                # only original images
                gt = gt.squeeze()
                tn = make_thumbnail_fullcut(img, gt, th_size, label=label)  # for full gland images
                thumbnails.append(tn)

                if i == 0:
                    imwrite(str(Path(FLAGS.save_dir).expanduser() / "thumbnail_test.png"), tn)

            labels.extend(label_batch)

            assert len(X) == len(thumbnails)
            assert len(X) == len(labels)

    # Save features
    model_name = "handcrafted"
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
    imwrite(str(save_file), spriteimage)
    save_file = Path(FLAGS.save_dir).expanduser() / (
            "thumbnails_" + model_name + ".npy")  # format for saving numpy data (not compressed)
    # Revert to RGB:
    thumbnails = thumbnails.astype(np.uint8)
    with open(str(str(save_file)), 'wb') as thumbnails_file:  # it uses pickle, hence need to open file in binary mode
        np.save(thumbnails_file, thumbnails)
    print("Saved {} ({}x{} thumbnails".format(*thumbnails.shape))

    labels = np.array(labels)
    # Save labels:
    save_file = Path(FLAGS.save_dir).expanduser() / ("labels_" + model_name + ".tsv")
    with open(str(save_file), 'w') as labels_file:
        np.savetxt(labels_file, labels, delimiter='\t')
    print(
        "Saved labels ({}) - {} tumour and {} gland".format(labels.size, np.sum(labels == 2), np.sum(labels == 3)))

    # Save paths
    save_file = Path(FLAGS.save_dir).expanduser() / ("paths_" + model_name + ".csv")
    with open(str(save_file), 'w') as paths_file:
        for item in paths:
            paths_file.write("{}\n".format(item))
    print("Saved paths ({})".format(len(paths)))

    print("Done!")


def thumbnails_size_check(str_size):
    size = int(str_size)
    if size > 100:
        warn("Max allowed thumbnail size is 100")
        size = 100
    return size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-sd', '--save_dir', default="/gpfs0/well/rittscher/users/achatrian/cancer_phenotype/Results")
    parser.add_argument('--thumbnail_size', default=64, type=thumbnails_size_check)
    parser.add_argument('--augment_num', type=int, default=0)
    parser.add_argument('--full_glands', type=str2bool, default='y')

    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/rittscher/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--workers', default=2, type=int, help='the number of workers to load the data')

    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)