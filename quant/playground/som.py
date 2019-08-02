import os
import glob
import deepdish
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sompy.sompy import SOMFactory
from sklearn.neighbors import NearestNeighbors
import imageio
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import matplotlib as mpl
from sklearn.externals import joblib
import argparse
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='som + rcc')
parser.add_argument('--file', default='../features_lab_16.h5')
parser.add_argument('--mapsize', default=20, type=int)
parser.add_argument('--save_folder', default='som_10x10_feature_lab_16', type=str)

args = parser.parse_args()


def sub_train_som(data, mapsize):
    sm = SOMFactory.build(data,
                          mapsize=mapsize,
                          normalization=None,
                          initialization='random',
                          lattice="rect")
    sm.train(n_job=1, verbose='info', train_rough_len=30, train_finetune_len=100)
    return sm


def train_som_para(data, mapsize, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    savename = os.path.join(save_folder, 'model.joblib')
    if not os.path.exists(savename):
        # SOM

        sm_set = Parallel(n_jobs=4)(delayed(sub_train_som)(data, mapsize) for _ in range(50))
        sm_idx = np.argmin([sm.calculate_topographic_error() for sm in sm_set])
        sm = sm_set[sm_idx]
        joblib.dump(sm, savename)

    else:
        sm = joblib.load(savename)

    return sm


def train_som(data, mapsize, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    savename = os.path.join(save_folder, 'model.joblib')
    if not os.path.exists(savename):
        # SOM
        sm = SOMFactory.build(data,
                              mapsize=mapsize,
                              normalization=None,
                              initialization='pca',
                              lattice="rect")
        sm.train(n_job=1, verbose='info', train_rough_len=30, train_finetune_len=100)
        joblib.dump(sm, savename)
    else:
        sm = joblib.load(savename)

    return sm


def som_visualisation(sm, X, X_names, cluster_idx=None):
    # define font
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 120)

    codebook = sm.codebook.matrix
    mapsize = sm.codebook.mapsize

    # examplar patches
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(codebook)
    distance, indices = nbrs.kneighbors(X)

    if np.any(cluster_idx):
        ############################################################
        best_matches = list()
        for idx in range(codebook.shape[0]):
            flag = indices == idx
            if np.any(flag):
                best_matches.append(np.array(X_names)[flag.ravel()][distance[flag].argmin()])
            else:
                best_matches.append(None)
        patches = best_matches

        patch_images = []
        colors = mpl.rcParams['axes.prop_cycle']()
        colors = [next(colors)['color'] for _ in range(len(np.unique(cluster_idx)))]

        for i_, (patch, icluster) in enumerate(zip(patches, cluster_idx)):
            if patch:
                img = imageio.imread(patch)
                seg = patch.replace('img', 'seg')
                try:
                    mask = imageio.imread(seg)
                except:
                    mask = np.ones((384, 384, 3), dtype=np.uint8)
            else:
                img = np.zeros((384, 384, 3), dtype=np.uint8)
                mask = np.zeros((384, 384, 3), dtype=np.uint8)

            img[mask == 0] = 0
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)

            draw.rectangle(((0, 0), (140, 100)), fill=colors[icluster])
            draw.text((0, 0), str(icluster + 1), (0, 0, 0), font=font)

            draw.text((0, 400), str(i_ + 1), (0, 0, 0), font=font)

            img = np.array(img)

            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            patch_images.append(img)

        patch_images = torch.cat(patch_images, dim=0)
        patch_images = make_grid(patch_images, nrow=mapsize[0])
        patch_images = patch_images.numpy()
        patch_images = patch_images.transpose(1, 2, 0)

    else:

        best_matches = list()
        for idx in range(codebook.shape[0]):
            flag = indices == idx
            if np.any(flag):
                best_matches.append(np.array(X_names)[flag.ravel()][distance[flag].argmin()])
            else:
                best_matches.append(None)
        patches = best_matches

        patch_images = []

        for i_, patch in enumerate(patches):
            if patch:
                img = imageio.imread(patch)
                seg = patch.replace('img', 'seg')
                try:
                    mask = imageio.imread(seg)
                except:
                    mask = np.ones((384, 384, 3), dtype=np.uint8)
            else:
                img = np.zeros((384, 384, 3), dtype=np.uint8)
                mask = np.zeros((384, 384, 3), dtype=np.uint8)

            img[mask == 0] = 0
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)

            draw.text((0, 400), str(i_), (0, 0, 0), font=font)

            img = np.array(img)

            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            patch_images.append(img)

        patch_images = torch.cat(patch_images, dim=0)
        patch_images = make_grid(patch_images, nrow=mapsize[0])
        patch_images = patch_images.numpy()
        patch_images = patch_images.transpose(1, 2, 0)

    return patch_images


######################################################
def main():

    # create folder to save clustering results
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # load a .h5 file containing features
    mat = deepdish.io.load(args.file)
    sampled_domain_output = mat['z']
    sampled_name = mat['name']

    X_transformed = sampled_domain_output
    sampled_name = [os.path.join('..', name) for name in sampled_name]
    ###########################################################################
    # training som
    sm = train_som(X_transformed, [args.mapsize, args.mapsize], save_folder)
    sm_grid_name = os.path.join(save_folder, 'som_grid.png')
    if not os.path.exists(sm_grid_name):
        sm_grid = som_visualisation(sm, X_transformed, sampled_name, cluster_idx=None)
        imageio.imwrite(sm_grid_name, sm_grid)

    ##########################################################################
    # markov clustering
    savename = os.path.join(save_folder, 'cluster_assignment.joblib')
    if not os.path.exists(savename):
        # umap
        import umap
        import markov_clustering as mc

        codebooks = sm.codebook.matrix

        best_Q = []
        all_clusters = []
        all_matrix = []
        all_neighbours = []
        for n_neighbour in range(10, 31, 5):
            print('n_neighbour %d' % n_neighbour)
            matrix = umap.umap_.fuzzy_simplicial_set(X=codebooks, n_neighbors=n_neighbour, random_state=np.random.RandomState(0),
                                                     metric='euclidean')

            all_inflation = []
            all_Q = []

                all_inflation.append(inflation)
                all_Q.append(Q)

            best_Q.append(np.max(all_Q))
            result = mc.run_mcl(matrix, inflation=all_inflation[np.argmax(all_Q)])  # run MCL with default parameters
            clusters = mc.get_clusters(result)
            all_clusters.append(clusters)
            all_matrix.append(matrix)
            all_neighbours.append(n_neighbour)
            # mc.draw_graph(matrix, clusters, pos=embedding, node_size=50, with_labels=False, edge_color="silver")

        clusters = all_clusters[np.argmax(best_Q)]
        matrix = all_matrix[np.argmax(best_Q)]
        n_neighbour = all_neighbours[np.argmax(best_Q)]
        embedding_umap = umap.UMAP(n_neighbors=n_neighbour,
                              min_dist=0.3,
                              metric='euclidean').fit_transform(codebooks)

        all_membership = np.vstack([[[x, idx] for x in cluster] for idx, cluster in enumerate(clusters)])
        sort_order = all_membership[:, 0].argsort()
        all_membership = all_membership[sort_order]
        P = all_membership[:, 1]

        mc.draw_graph(matrix, clusters, pos=embedding_umap, node_size=50, with_labels=False, edge_color="silver")
        plt.savefig(os.path.join(save_folder, 'mcl.png'))
        plt.close()

        joblib.dump({'embedding_umap': embedding_umap, 'matrix': matrix, 'clusters': clusters, 'P': P}, savename)
    else:
        P = joblib.load(savename)['P']

    savename = os.path.join(save_folder, 'sm_grid_label.png')
    if not os.path.exists(savename):
        sm_grid_label = som_visualisation(sm, X_transformed, sampled_name, cluster_idx=P)
        imageio.imwrite(savename, sm_grid_label)

    ############################################################################
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sm.codebook.matrix)
    distance, indices = nbrs.kneighbors(X_transformed)
    distance = distance.ravel()
    indices = indices.ravel()

    # examplar patches
    for idx in np.unique(P):
        savename = os.path.join(save_folder, 'cluster_' + str(idx) + '.png')
        if not os.path.exists(savename):
            selected_codebook_indices = np.where(P == idx)[0]

            best_matches = []
            keep_codebook_idx = []
            for codebook_idx in selected_codebook_indices:
                flag = indices == codebook_idx
                smallest_distance_indices = distance[flag].argsort()
                selected_names = np.array(sampled_name)[flag][smallest_distance_indices][:8].ravel()
                best_matches.append(selected_names)
                keep_codebook_idx.append([codebook_idx] * len(selected_names))

            best_matches = np.hstack(best_matches)

            patches = best_matches
            keep_codebook_idx = np.hstack(keep_codebook_idx)

            patch_images = []
            for patch, codebook_idx in zip(patches, keep_codebook_idx):
                if patch:
                    img = imageio.imread(patch)

                    img = Image.fromarray(img)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 120)
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 400), str(codebook_idx), (0, 0, 0), font=font)
                    img = np.array(img)

                else:
                    img = np.zeros((512, 512, 3), dtype=np.uint8)

                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                patch_images.append(img)

            patch_images = torch.cat(patch_images, dim=0)
            patch_images = make_grid(patch_images, nrow=8)
            patch_images = patch_images.numpy()
            patch_images = patch_images.transpose(1, 2, 0)

            imageio.imwrite(savename, patch_images)


def cluster_visualisation_group(codebook, X, X_names, idx):
    # examplar patches
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(codebook)
    distance, indices = nbrs.kneighbors(X)

    flag = indices == idx
    names = np.array(X_names)[flag.ravel()]
    distances = distance[flag]

    sort_order = np.argsort(distances)
    names = names[sort_order]

    patches = names
    patch_images = []

    nimages = 64
    if len(patches) < 64:
        nimages = len(patches)

    for i in range(nimages):
        patch = patches[i]
        img = imageio.imread(patch)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        patch_images.append(img)

    patch_images = torch.cat(patch_images, dim=0)
    patch_images = make_grid(patch_images, nrow=8)
    patch_images = patch_images.numpy()
    patch_images = patch_images.transpose(1, 2, 0)

    return patch_images


if __name__ == "__main__":
    main()
