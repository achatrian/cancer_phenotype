import sys
import argparse
from warnings import warn
from pathlib import Path
import time
import multiprocessing as mp

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
    sys.path.append("/users/rittscher/achatrian/cancer_phenotype/segment")
    sys.path.append("/users/rittscher/achatrian/cancer_phenotype/phenotyping")
    sys.path.append("/users/rittscher/achatrian/cancer_phenotype/ck5")
else:
    sys.path.append("/Users/andreachatrian/Documents/Repositories/cancer_phenotype")

from segment.utils import check_mkdir
from repo_old.phenotyping import feature_extraction, \
    make_thumbnail_fullcut, create_sprite_image, augment_glands
from repo_old.phenotyping import GlandDataset

exp_name = ''


class FeatureSummariser(mp.Process):

    def __init__(self, id, input_queue, output_queue, thumbnail_size, augment_num):
        #################################
        mp.Process.__init__(self, name='FeatureSummariser')
        self.daemon = True  # required (should read about it)
        self.id = id
        #################################
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.th_size = (thumbnail_size, ) * 2
        self.N = augment_num

    def run(self):
        count = 0
        print("[p{}] starting".format(self.id))

        while True:
            data = self.input_queue.get()

            if data is None:
                self.input_queue.task_done()
                self.output_queue.put("done")
                break  # exit from infinite  te loop

            inputs = data[0]
            gts = data[1]
            colours_n_size = data[2].numpy()
            label_batch = data[3]
            label_batch = list(label_batch.numpy())
            M = inputs.shape[0]  # remember how many different gland images are there

            paths = data[4]

            assert np.any(np.array(inputs.shape))  # non empty data

            imgs = inputs.numpy().transpose(0, 2, 3, 1)
            gts = gts.numpy().transpose(0, 2, 3, 1)
            imgs, gts = augment_glands(imgs, gts, N=self.N)  # consecutive images are augmented versions of first image

            handcraft_feats = feature_extraction(imgs, gts)

            handcraft_feats = handcraft_feats.reshape((handcraft_feats.shape[0] // (self.N + 1), (self.N + 1), -1))
            handcraft_feats = handcraft_feats.mean(axis=1)

            assert handcraft_feats.shape[0] == M  # ensure there is one feature vector per image

            bad_imgs = []
            X_chunk = []
            tmb_chunk = []
            #for j, (gl_features, gl_alg_features) in enumerate(zip(glands_features, alg_feats)):
            for j, gl_features in enumerate(handcraft_feats):
                # Checks FIXME should not have to do this
                X_chunk.append(gl_features)  # feature vector
                X_chunk[-1] = np.concatenate((gl_features, colours_n_size[j, ...]))

            # Save original images
            gts = gts[0::(self.N + 1), ...]
            for j, (img, gt, label) in enumerate(zip(imgs[0::(self.N + 1), ...], gts, label_batch)):
                # only original images
                gt = gt.squeeze()
                tn = make_thumbnail_fullcut(img, gt, self.th_size, label=label)  # for full gland images
                tmb_chunk.append(tn)

                if j == 0:
                    imwrite(str(Path(FLAGS.save_dir).expanduser() / "thumbnail_test.png"), tn)

            assert len(X_chunk) == len(tmb_chunk)
            assert len(X_chunk) == len(label_batch)

            # Save original images
            print("[p{}] saving thumbnails".format(self.id))
            # Save original images
            gts = gts[0::(self.N + 1), ...]
            for j, (img, gt, label) in enumerate(zip(imgs[0::(self.N + 1), ...], gts, label_batch)):
                # only original images
                gt = gt.squeeze()
                tn = make_thumbnail_fullcut(img, gt, self.th_size, label=label)  # for full gland images
                tmb_chunk.append(tn)

            count += 1
            print("[p{}] processed {} batches".format(self.id, count))
            self.output_queue.put([X_chunk, tmb_chunk, label_batch, paths])
            self.input_queue.task_done()


def main(FLAGS):

    datasets = [GlandDataset(FLAGS.data_dir, mode, 299) for mode in ['train', 'val', 'test']]
    loaders = [DataLoader(ds, batch_size=FLAGS.batch_size, shuffle=False, num_workers = 2 if FLAGS.workers > 1 else 1) for ds in datasets]

    tqdm.write("Generating features and thumbnails ...")

    INPUT_QUEUE_SIZE = 2 * FLAGS.workers
    input_queue = mp.JoinableQueue(maxsize=INPUT_QUEUE_SIZE)
    output_queue = mp.JoinableQueue(maxsize=10000)
    print("Spawning processes ...")
    start_time = time.time()

    X = []  # feature matrix
    thumbnails = []
    labels = []
    check_mkdir(FLAGS.save_dir)
    paths = []

    running = max(FLAGS.workers - 2, 1)
    for i in range(running):
        FeatureSummariser(i, input_queue, output_queue, FLAGS.thumbnail_size, FLAGS.augment_num).start()

    for loader, mode in zip(loaders, ['train', 'val', 'test']):
        print("[main] loading {} data".format(mode))
        # Use train test and validate datasets
        for i, data in enumerate(loader):
            if not output_queue.empty():
                print("[main] getting chunks {}/{} ({} elapsed)".format(i, len(loader), time.time() - start_time))
                processed = output_queue.get_nowait()
                if processed == "done":
                    running -= 1
                else:
                    x_chunk, tns_chunk, lbls_chunk, paths_chunk = processed
                    X.extend(x_chunk)
                    thumbnails.extend(tns_chunk)
                    labels.extend(lbls_chunk)
                    paths.extend(paths_chunk)
                output_queue.task_done()
            input_queue.put(data)

    print("[main] sending end signals")
    for i in range(FLAGS.workers):
        input_queue.put(None)

    input_queue.join()
    while bool(running):
        processed = output_queue.get()
        if processed == "done":
            running -= 1
            print("Still running = {}".format(running))
        else:
            x_chunk, tns_chunk, lbls_chunk, paths_chunk = processed
            X.extend(x_chunk)
            thumbnails.extend(tns_chunk)
            labels.extend(lbls_chunk)
            paths.extend(paths_chunk)
        output_queue.task_done()

    output_queue.join()

    # Save features
    model_name = "handcrafted"
    X = np.array(X)
    header = "handcrafted_feats"
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

    # Save labels:
    labels = np.array(labels)
    save_file = Path(FLAGS.save_dir).expanduser() / ("labels_" + model_name + ".tsv")
    with open(str(save_file), 'w') as labels_file:
        np.savetxt(labels_file, labels, delimiter='\t')
    print("Saved labels ({}) - {} tumour and {} gland".format(labels.size, np.sum(labels == 2), np.sum(labels == 3)))
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

    parser.add_argument('-sd', '--save_dir', default="/gpfs0/well/rittscher/users/achatrian/cancer_phenotype/Results")
    parser.add_argument('--thumbnail_size', default=64, type=thumbnails_size_check)
    parser.add_argument('--augment_num', type=int, default=0)

    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('-d', '--data_dir', type=str, default="/gpfs0/well/rittscher/users/achatrian/cancer_phenotype/Dataset")
    parser.add_argument('--workers', default=4, type=int, help='the number of workers to load the data')

    parser.add_argument('--num_class', type=int, default=1)
    parser.add_argument('-nf', '--num_filters', type=int, default=64, help='mcd number of filters for unet conv layers')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)