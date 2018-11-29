import os
import sys
import argparse
import numpy as np
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import time

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler



def main(FLAGS):

    pca_filepath = FLAGS.save_dir + "/pcaed_feats_{}.txt".format(str(Path(FLAGS.features_file).name)[:-4])
    X_pca = np.loadtxt(pca_filepath, delimiter=' ')
    with open(str(Path(FLAGS.save_dir)/"pca_obj_{}.pkl".format(str(Path(FLAGS.features_file).name)[:-4])), 'rb') as pca_obj_file:
        pca = pickle.load(pca_obj_file)

    # Check X:
    good = np.all(np.isfinite(X_pca))
    if not good:
        print("Correcting for X's NaNs")
        X_pca[np.isnan(X_pca)] = 0.0

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cum_var)
    ninetynine = np.where(cum_var > 0.99)[0][0]
    print("Ninetynine = {}".format(ninetynine))

    # Preprocessing
    X_pca = StandardScaler().fit_transform(X_pca)

    num_feats = ninetynine
    X_white = X_pca[:, :num_feats]

    # For t-SNE, PCA is recommended as a first step to reduce dimensionality
    run_time = time.time()
    whiten = False
    ica = FastICA(n_components=ninetynine, whiten=whiten, max_iter=400)  # assuming components are already whitened!
    # !!! it is possible to generative negative eigenvalue and make the algorithm fail with large matrix that has determinant close to zero
    # for ICA, should reduce the number of comoponents to number (rule of thumb, as many as 99% variance explanation in PCA)
    ica = ica.fit(X_pca)
    X_ica = ica.transform(X_white)
    run_time = time.time() - run_time
    print("X_ica's shape is ", X_ica.shape)
    print("Run in {}s".format(run_time))

    with open(FLAGS.FLAGS.save_dir + "/icaed_feats_{}.txt".format(str(FLAGS.features_file)[:-4]), 'w') as ica_feats_file:
        np.savetxt(ica_feats_file, X_ica, delimiter=' ')

    with open(str(Path(FLAGS.save_dir) / "ica_obj_{}.pkl".format(str(Path(FLAGS.features_file).name)[:-4])), 'wb') as ica_obj_file:
        pickle.dump(ica, ica_obj_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features_file', required=True)
    parser.add_argument('-lf', '--labels_file', default="")
    parser.add_argument('-sd', '--save_dir', default="/well/rittscher/users/achatrian/cancer_phenotype/Results")
    parser.add_argument('-r1', '--pca_reduce', default=2/3, type=float)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)