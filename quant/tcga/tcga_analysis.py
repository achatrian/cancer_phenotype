from pathlib import Path
import argparse
import warnings
from datetime import datetime
import json
import numpy as np
import joblib as jl
import pandas as pd
from tqdm import tqdm
import imageio
from sklearn.ensemble import IsolationForest
from sompy.sompy import SOMFactory
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import h5py
import umap
import markov_clustering as mc
from experiment.clustering_ import Clustering
from base.utils.utils import bytes2human


def save_log(log, data_dir):
    (data_dir/'data'/'logs').mkdir(exist_ok=True, parents=True)
    with (data_dir/'data'/'logs'/f'tcga_analysis_{log["date"]}.json').open('w') as log_file:
        json.dump(log, log_file)


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
    parser.add_argument('--isolation_contamination', default='auto', help="Contamination parameter in the isolation  (auto or float)")
    parser.add_argument('--map_size', type=int, default=100, help="Size of the self-organizing map on our dataset")
    parser.add_argument('--som_rough_epochs', type=int, default=100, help="Number of epochs to train som's for")
    parser.add_argument('--som_finetune_epochs', type=int, default=60)
    parser.add_argument('--mcl_pruning_frequency', type=int, default=1, help="Perform pruning every 'pruning_frequency' iterations.")
    parser.add_argument('--overwrite', action='store_true', help="Whether to discard old models and overwrite them")
    args = parser.parse_args()
    if args.isolation_contamination.isnumeric():
        args.isolation_contamination = float(args.isolation_contamination)
    save_dir = args.data_dir/'data'/'experiments'/'som'
    save_dir.mkdir(exist_ok=True, parents=True)
    log = {'date': str(datetime.now())}
    results_file = h5py.File(save_dir/'results.h5', mode='w')
    args_container = {key: (value if type(value) in {str, int, float, bool} else str(value))for key, value in vars(args).items()}
    results_group = results_file.create_group(json.dumps(args_container))  # store args
    # data clean-up; remove outlierss
    features_file = args.data_dir/'data'/'features'/args.experiment_name/args.features_filename
    try:
        features = pd.read_hdf(features_file.parent/'single_key.h5', 'features')
    except FileNotFoundError:
        slide_ids = list(h5py.File(features_file, 'r').keys())  # read keys to access different stored frames
        feature_frames = []
        for slide_id in tqdm(slide_ids, desc=f"Reading features for {len(slide_ids)} slides ..."):
            feature_frame = pd.read_hdf(features_file, slide_id)
            slide_id = slide_id[:-1]  # BUG SOMEWHERE INCLUDED A PERIOD AT THE END OF KEYS -- remove
            # form multiindex with both bounding box and slide id information
            feature_frame.index = pd.MultiIndex.from_arrays(((slide_id,)*len(feature_frame), feature_frame.index),
                                                            names=('slide_id', 'bounding_box'))
            feature_frames.append(feature_frame)
        features = pd.concat(feature_frames)
        features.to_hdf(features_file.parent/'single_key.h5', 'features')
    print(f"Features were read (size: {bytes2human(features.memory_usage().sum())})")
    print(f"Data shape: {features.shape}")
    log.update({'features_size': bytes2human(features.memory_usage().sum()), 'feature_frame_shape': features.shape})
    save_log(log, args.data_dir)
    if args.outlier_removal:
        print("Removing outliers ...")
        original_n = features.shape[0]
        c = Clustering('', x=features,
                       outlier_removal=IsolationForest(contamination=args.isolation_contamination, behaviour='new'))
        c.remove_outliers()
        features = c.x
        print(f"{original_n - features.shape[0]} outliers were removed.")
        log.update({'outlier_removal': True, 'num_removed_outliers': original_n - features.shape[0]})
    else:
        log.update({'outlier_removal': False, 'num_removed_outliers': 0})
    results_group.create_dataset('features', data=np.array(features))  # TODO - test
    save_log(log, args.data_dir)
    # train self-organizing map
    sm = train_som(features, (args.map_size, args.map_size), save_dir, args.som_rough_epochs, args.som_finetune_epochs)
    print(f"Codebook shape: {sm.codebook.matrix.shape}")
    results_group.create_dataset('codebook_matrix', data=np.array(sm.codebook.matrix))
    log.update({'som_codebook_shape': sm.codebook.matrix.shape})
    # cluster codebooks using umap
    save_path = save_dir/f'cluster_assignment_s{args.map_size}_er{args.som_rough_epochs}_ef{args.som_finetune_epochs}.joblib'
    fl_ = lambda r: tuple(float(i) for i in r) if type(r) in (tuple, list) else float(r)
    if args.overwrite or not save_path.exists():
        codebooks = sm.codebook.matrix
        best_Q = []
        all_clusters = []
        all_matrix = []
        all_neighbours = []
        upper_limit = 21
        for n_neighbour in range(700, 2101, 300):
            print('n_neighbour %d' % n_neighbour)
            with warnings.catch_warnings():  # ignore numba compile warnings
                warnings.simplefilter("ignore")
                matrix = umap.umap_.fuzzy_simplicial_set(X=codebooks, n_neighbors=n_neighbour,
                                                         random_state=np.random.RandomState(0),
                                                         metric='euclidean')

            all_inflation = []
            all_Q = []
            for inflation in tqdm([i / 10 for i in range(11, 33, 5)]):
                iteration_group = results_group.create_group(f'n_neighbours:{n_neighbour}_inflation:{inflation}')
                tqdm.write(f"Running markov clustering for {n_neighbour} neighbors and inflation={inflation}")
                result = mc.run_mcl(matrix, inflation=inflation, pruning_frequency=args.mcl_pruning_frequency,
                                    verbose=True)  # NB mcl.prune breaks with scipy>0.13, see 04/10/2019 log
                iteration_group.create_dataset('result_matrix', data=result.toarray())
                clusters = mc.get_clusters(result)
                for i, cluster_nodes in enumerate(clusters):
                    iteration_group.create_dataset(f'cluster{i}', data=np.array(cluster_nodes))
                iteration_group.create_dataset('num_clusters', data=len(clusters))
                tqdm.write(f"Number of clusters: {len(clusters)}")
                tqdm.write(f"Computing modularity for inflation={inflation} ...")
                Q = mc.modularity(matrix=result, clusters=clusters)  # this step is very long for low inflation
                iteration_group.create_dataset('modularity', data=Q)
                tqdm.write(f"Modularity for inflation={inflation} is {Q}.")
                all_inflation.append(inflation)
                all_Q.append(Q)
            best_Q.append(np.max(all_Q))
            print(f"Best Q value = {np.max(all_Q)} for inflation={all_inflation[np.argmax(all_Q)]}")
            log.update({f'{n_neighbour}_neighbour': {'inflations': fl_(all_inflation), 'Qs': fl_(all_Q)}})
            save_log(log, args.data_dir)
            result = mc.run_mcl(matrix, inflation=all_inflation[np.argmax(all_Q)])  # run MCL with default parameters
            clusters = mc.get_clusters(result)
            all_clusters.append(clusters)
            all_matrix.append(matrix)
            all_neighbours.append(n_neighbour)
            # mc.draw_graph(matrix, clusters, pos=embedding, node_size=50, with_labels=False, edge_color="silver")
        clusters = all_clusters[np.argmax(best_Q)]
        matrix = all_matrix[np.argmax(best_Q)]
        n_neighbour = all_neighbours[np.argmax(best_Q)]
        with warnings.catch_warnings():  # ignore numba compile warnings
            warnings.simplefilter("ignore")
            embedding_umap = umap.UMAP(n_neighbors=n_neighbour,
                                       min_dist=0.3,
                                       metric='euclidean').fit_transform(codebooks)
        all_membership = np.vstack([[[x, idx] for x in cluster] for idx, cluster in enumerate(clusters)])
        sort_order = all_membership[:, 0].argsort()
        all_membership = all_membership[sort_order]
        P = all_membership[:, 1]
        results_group.create_dataset('membership_index', data=P)
        results_group.create_dataset('final_num_clusters', data=np.max(P))
        mc.draw_graph(matrix, clusters, pos=embedding_umap, node_size=50, with_labels=False, edge_color="silver")
        plt.savefig(Path(save_dir, 'mcl.png'))  # not working on notebook
        results_group.create_dataset('mcl_viz', data=imageio.imread(Path(save_dir, 'mcl.png')))
        plt.close()
        jl.dump({'embedding_umap': embedding_umap, 'matrix': matrix, 'clusters': clusters, 'P': P}, save_path)
    else:
        P = jl.load(save_path)['P']
    print(f"Number of clusters: {np.max(P)}")
    log['num_clusters'] = int(np.max(P))
    save_log(log, args.data_dir)
    # assign each data-point to a code vector
    nearest_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sm.codebook.matrix)
    distance, indices = nearest_neighbours.kneighbors(features)  # find nearest SOM cluster for each data-point
    distance = distance.ravel()
    indices = indices.ravel()
    # assign each point to the markov cluster of his nearest codebook
    cluster_assignment = pd.Series(data=[P[idx] for idx in indices], index=features.index)
    plt.hist(cluster_assignment)
    plt.savefig(save_dir/f'cluster_assignment_s{args.map_size}_er{args.som_rough_epochs}_ef{args.som_finetune_epochs}.png')
    plt.close()
    cluster_assignment.to_csv(save_dir/'cluster_assignment.csv')
    # gather examples from each cluster
    membership_numbers = [np.sum(cluster_assignment == p) for p in np.unique(P)]
    clusters = list(p for p in np.unique(P) if membership_numbers[p] > 10)  # exclude very small clusters
    c = Clustering('markov', x=features, y=cluster_assignment, clusters=clusters)
    cluster_centers = tuple(sm.codebook.matrix[np.where(P == i)[0]] for i in clusters)
    examples = c.get_examples(args.data_dir, n_examples=10, cluster_centers=cluster_centers, image_dim_increase=0.25)
    c.save_examples_grid(save_dir, examples, image_size=512)
    results_file.close()




