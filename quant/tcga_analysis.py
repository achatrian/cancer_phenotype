from pathlib import Path
import argparse
import numpy as np
import joblib as jl
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sompy.sompy import SOMFactory
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import h5py
import umap
import markov_clustering as mc
from quant.experiment.clustering import Clustering
from base.utils.utils import bytes2human


def sub_train_som(data, mapsize):
    sm = SOMFactory.build(data,
                          mapsize=mapsize,
                          normalization=None,
                          initialization='random',
                          lattice="rect")
    sm.train(n_job=1, verbose='info', train_rough_len=30, train_finetune_len=100)
    return sm


def train_som_para(data, mapsize, save_dir, overwrite=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir/'model.joblib'
    if overwrite or not save_path.exists():
        # SOM
        sm_set = jl.Parallel(n_jobs=4)(jl.delayed(sub_train_som)(data, mapsize) for _ in range(50))
        sm_idx = np.argmin([sm.calculate_topographic_error() for sm in sm_set])
        sm = sm_set[sm_idx]
        jl.dump(sm, save_path)

    else:
        sm = jl.load(save_path)
    return sm


def train_som(data, mapsize, save_dir, train_rough_len=30, train_finetune_len=100, overwrite=False):
    # sequential version of above
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir/'model.joblib'
    if overwrite or not save_path.exists():        # SOM
        sm = SOMFactory.build(data,
                              mapsize=mapsize,
                              normalization=None,
                              initialization='pca',
                              lattice="rect")
        sm.train(n_job=1, verbose='info', train_rough_len=train_rough_len, train_finetune_len=train_finetune_len)
        jl.dump(sm, save_path)
    else:
        sm = jl.load(save_path)
    return sm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--features_filename', type=str, default='filtered.h5', help="Basename of files storing features")
    parser.add_argument('--outlier_removal', action='store_true')
    parser.add_argument('--isolation_contamination', default='auto', help="Contamination parameter in the isolation forest")
    parser.add_argument('--map_size', type=int, default=100, help="Size of the self-organizing map on our dataset")
    parser.add_argument('--som_epochs', type=int, default=90, help="Number of epochs to train som's for")
    parser.add_argument('--overwrite', action='store_true', help="Whether to discard old models and overwrite them")
    args = parser.parse_args()
    save_dir = args.data_dir/'data'/'experiments'/'som'
    save_dir.mkdir(exist_ok=True, parents=True)
    # data clean-up; remove outlierss
    features_file = args.data_dir/'data'/'features'/args.experiment_name/args.features_filename
    try:
        features = pd.read_hdf(features_file.parent/'single_key.h5', 'features')
    except FileNotFoundError:
        slide_ids = list(h5py.File(features_file, 'r').keys())  # read keys to access different stored frames
        feature_frames = []
        for slide_id in tqdm(slide_ids, desc=f"Reading features for {len(slide_ids)} slides ..."):
            feature_frame = pd.read_hdf(features_file, slide_id)
            # form multiindex with both bounding box and slide id information
            feature_frame.index = pd.MultiIndex.from_arrays(((slide_id,)*len(feature_frame), feature_frame.index),
                                                            names=('slide_id', 'bounding_box'))
            feature_frames.append(feature_frame)
        features = pd.concat(feature_frames)
        features.to_hdf(features_file.parent/'single_key.h5', 'features')
    print(f"Features were read (size: {bytes2human(features.memory_usage().sum())})")
    print(f"Data shape: {features.shape}")
    if args.outlier_removal:
        print("Removing outliers ...")
        original_n = features.shape[0]
        c = Clustering('', x=features,
                       outlier_removal=IsolationForest(contamination=args.isolation_contamination, behaviour='new'))
        c.remove_outliers()
        features = c.x
        print(f"{original_n - features.shape[0]} outliers were removed.")
    # train self-organizing map
    sm = train_som(features, (args.map_size, args.map_size), save_dir, args.som_epochs, 50)
    print(f"Codebook shape: {sm.codebook.matrix.shape}")
    # cluster using umap
    save_folder = '/well/rittscher/users/achatrian/temp'
    save_path = Path(save_folder)/f'cluster_assignment_s{args.map_size}_e{args.som_epochs}.joblib'
    if not save_path.exists():
        codebooks = sm.codebook.matrix
        best_Q = []
        all_clusters = []
        all_matrix = []
        all_neighbours = []
        upper_limit = 21
        for n_neighbour in [30]:  # range(10, 31, 5):
            print('n_neighbour %d' % n_neighbour)
            matrix = umap.umap_.fuzzy_simplicial_set(X=codebooks, n_neighbors=n_neighbour,
                                                     random_state=np.random.RandomState(0),
                                                     metric='euclidean')
            all_inflation = []
            all_Q = []
            for inflation in tqdm([i / 10 for i in range(11, 30)]):
                result = mc.run_mcl(matrix, inflation=inflation)
                clusters = mc.get_clusters(result)
                Q = mc.modularity(matrix=result, clusters=clusters)
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
        plt.savefig(Path(save_folder, 'mcl.png'))  # not working on notebook
        plt.close()
        jl.dump({'embedding_umap': embedding_umap, 'matrix': matrix, 'clusters': clusters, 'P': P}, save_path)
    else:
        P = jl.load(save_path)['P']
    print(f"Number of clusters: {np.max(P)}")
    # assign each data-point to a code vector
    nearest_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sm.codebook.matrix)
    distance, indices = nearest_neighbours.kneighbors(features)  # find nearest SOM cluster for each data-point
    distance = distance.ravel()
    indices = indices.ravel()
    # assign each point to the markov cluster of his nearest codebook
    cluster_assignment = pd.Series(data=[P[idx] for idx in indices], index=features.index)
    plt.hist(cluster_assignment)
    plt.savefig(save_dir/f'cluster_assignment_s{args.map_size}_e{args.som_epochs}.png')
    plt.close()
    # gather examples from each cluster
    membership_numbers = [np.sum(cluster_assignment == p) for p in np.unique(P)]
    clusters = list(p for p in np.unique(P) if membership_numbers[p] > 10)  # exclude very small clusters
    cluster_centers = []
    for i in c.clusters:
        cluster_centers.append(
            sm.codebook.matrix[np.where(P == i)[0]]
        )
    c = Clustering('markov', x=features, y=cluster_assignment, clusters=clusters)
    cluster_centers = tuple(sm.codebook.matrix[np.where(P == i)[0]] for i in clusters)
    examples = c.get_examples(args.data_dir, n_examples=10, cluster_centers=cluster_centers, image_dim_increase=0.25)
    c.save_examples_grid('/well/rittscher/users/achatrian/temp', examples, image_size=512)




