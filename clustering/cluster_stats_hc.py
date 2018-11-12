import os, sys
import argparse
import numpy as np
from pathlib import Path
from itertools import product
import multiprocessing as mp

import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from scipy.stats import mode

from tensorboardX import SummaryWriter
import imageio
import cv2
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import DBSCAN, AffinityPropagation, AgglomerativeClustering, KMeans
#from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.feature_selection import SelectKBest, chi2

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import NearestNeighbors

import pickle

from scipy.cluster import vq

import csv
import datetime


def cluster_query(X, labels, ks):
    nbrs = NearestNeighbors(100, radius=1.0, algorithm='ball_tree').fit(X)
    unique_labels = np.unique(labels)
    fitness_k = []
    inpoints_k = []
    for k in ks:
        print("run for k = {}".format(k))
        distances, indices = nbrs.kneighbors(X, k)  # k nearest neighbours
        inpoints = []
        ones_own = []
        for neighs, point_label in zip(indices, labels):
            neighs_labels = labels[neighs]
            inpoints.append([neighs_labels == c for c in unique_labels])
            ones_own.append(np.sum(neighs_labels == point_label))
        ones_own = np.array(ones_own)
        feature_fitness = [np.sum(ones_own[labels == c]) / (k * np.sum(labels == c)) for c in unique_labels]
        fitness_k.append(feature_fitness)
        inpoints_k.append(inpoints)
    return fitness_k, unique_labels

### Args ###:
features_file = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/handcrafted/feats_handcrafted.csv"
thumbnails_file = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/handcrafted/thumbnails_handcrafted.npy"
sprite_file = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/handcrafted/sprite_handcrafted.png"
labels_file = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/handcrafted/labels_handcrafted.tsv"
paths_file = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/handcrafted/paths_handcrafted.tsv"
patients_file = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/EUSlides_samples.csv"
save_dir = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/handcrafted"
log_dir = str(Path(features_file).parent)
thumbnail_size = 64
pca_reduce = 2/3
###

load_pca = True
if load_pca:
    pca_filepath = save_dir + "/pcaed_feats_{}.txt".format(str(Path(features_file).name)[:-4])
    X_pca = np.loadtxt(pca_filepath, delimiter=' ')
    with open(str(Path(save_dir)/"pca_obj_{}.pkl".format(str(Path(features_file).name)[:-4])), 'rb') as pca_obj_file:
        pca = pickle.load(pca_obj_file)
    print("Loaded pca'ed X at {}".format(str(datetime.datetime.now())))
else:
    with open(features_file, 'r') as feats_file:
        X = np.loadtxt(feats_file, skiprows=1)
        # need to skip header row (?)
        print("Loaded X at {}".format(str(datetime.datetime.now())))

labels = np.loadtxt(labels_file, delimiter='\t')
print("Loaded labels at {}".format(str(datetime.datetime.now())))

# feature_selection = False
# if feature_selection:
#     X_new = SelectKBest(chi2, k=num_feats).fit_transform(X_pca + abs(X_pca.min()), labels)
#     X_pca = StandardScaler().fit_transform(X_new)

sns.set(style="darkgrid")
fitnesses_numfeats = dict()
k_range = np.arange(2, 20, 2)
num_feats = 200

# Query clusters
fitness_k, unique_labels = cluster_query(X_pca[:, :num_feats], labels, k_range)
fitness_k = np.array(fitness_k)
plt.plot(np.tile(k_range[:, np.newaxis], (1, 2)), fitness_k)
plt.title("{} feats".format(num_feats))
plt.show(block=True)

with open(os.path.join(save_dir, "query_hc.csv"), 'w') as query_file:
    np.savetxt(query_file, fitness_k, header=str(k_range))

X_pca = X_pca[-1000:, :]

do_ICA = True  # converges for unet
ica_filepath = save_dir + "/icaed_feats_{}.txt".format(str(Path(features_file).name)[:-4])
X_ica = np.loadtxt(ica_filepath, delimiter=' ')
with open(str(Path(save_dir)/"ica_obj_{}.pkl".format(str(Path(features_file).name)[:-4])), 'rb') as ica_obj_file:
    ica = pickle.load(ica_obj_file)

# Query clusters
fitness_k, unique_labels = cluster_query(X_ica, labels, k_range)
fitness_k = np.array(fitness_k)
plt.plot(np.tile(k_range[:, np.newaxis], (1, 2)), fitness_k)
plt.title("{} feats".format(num_feats))
plt.show(block=True)

