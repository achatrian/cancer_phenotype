import os, sys
import argparse
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from tensorboardX import SummaryWriter

#import tensorflow as tf
#from tensorflow.contrib.tensorboard.plugins import projector


def main(FLAGS):

    ### Preprocessing and visualization ###

    with open(FLAGS.features_file, 'r') as feature_file:
        X = np.loadtxt(feature_file, skiprows=1)  # need to skip header row (?)

    # For t-SNE, PCA is recommended as a first step to reduce dimensionality
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=int(X.shape[1]*FLAGS.pca_reduce))  # reduce number of features to two thirds of originals
    pca = pca.fit(X)
    X_new = pca.transform(X)

    #sns.set(style="darkgrid")
    #plt.plot(pca.explained_variance_ratio_)

    # t-SNE with tensorboard
    log_dir = Path(FLAGS.features_file).parent
    #os.mkdir(log_dir/"projector")

    with open(FLAGS.thumbnail_file, 'rb') as thumbnail_file:
        thumbnails = np.load(thumbnail_file)

    writer = SummaryWriter(log_dir=str(log_dir))
    X_t = torch.from_numpy(X_new)
    thumbnails_t = torch.from_numpy(thumbnails).permute(0, 3, 1, 2).type(torch.float32)  # turn to NCHW for tensorboardX embedding
    writer.add_embedding(X_t, label_img=thumbnails_t)

    ### Clustering ###








    """
    tf_X = tf.Variable(X_new, trainable=False)
    saver = tf.train.Saver([tf_X])

    # based on https://www.easy-tensorflow.com/tf-tutorials/tensorboard/tb-embedding-visualization
    # http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/

    th_size = (FLAGS.thumbnail_size,) * 2
    with tf.Session() as sess:

        sess.run(tf_X.initializer)

        writer = tf.summary.FileWriter(log_dir /"projector", sess.graph)

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embeddingding:0'
        embedding.metadata_path = log_dir/"projector"/"metadata.tsv"  # TODO I don't have labels so no metadata is necessary?
        embedding.sprite.image_path = log_dir/"projector"/FLAGS.sprite_file
        embedding.sprite.single_img_dim.extend(th_size)

        projector.visualize_embeddings(summary_writer=writer, config=config)
    """






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--features_file', required=True)
    #parser.add_argument('-if', '--sprite_file', required=True)
    parser.add_argument('-tf', '--thumbnail_file', required=True)
    parser.add_argument('--thumbnail_size', default=64, type=int)

    parser.add_argument('-r1', '--pca_reduce', default=1/10, type=float)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)