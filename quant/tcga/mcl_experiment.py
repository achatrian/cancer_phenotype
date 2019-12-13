from pathlib import Path
import json
from datetime import datetime
from collections import namedtuple
import warnings
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import joblib as jl
import pandas as pd
from tqdm import tqdm
import imageio
from sklearn.ensemble import IsolationForest
from sompy.sompy import SOMFactory
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, \
    adjusted_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.cluster import KMeans
import h5py
import umap
import markov_clustering as mc
from base.utils.utils import bytes2human
from quant.experiment import BaseExperiment  # should inherit from clustering instead ?
from quant.utils import read_parameter_values
from quant.viz import get_cluster_examples, make_cluster_grids
import logging  # disable annoying font messagess

logging.getLogger('matplotlib.font_manager').disabled = True


class MCLExperiment(BaseExperiment):
    parameters_names = ('num_neighbors', 'inflation')
    parameters_defaults = ('2000', '1.1')
    parameters_descriptions = (
        "Number of neighbours in markov clustering",
        "Inflation exponent in markov clustering"
    )

    preprocess_parameters_names = ('rough_epochs', 'finetune_epochs', 'map_size')
    preprocess_parameters_defaults = ('100', '60', '100')
    preprocess_parameters_descriptions = (
        "Number of epochs to train som's for (rough adjustment)",
        "Number of epochs to train som's for (fine tuning adjustment)",
        "Size of the self-organizing map"
    )

    def __init__(self, args):
        super().__init__(args)
        self.features, self.inliers = (None,) * 2
        self.save_dir = self.args.data_dir / 'data' / 'experiments' / self.name().lower().split('experiment')[0]
        self.save_dir.mkdir(exist_ok=True, parents=True)
        # use json string with parameters values as experiment key
        parameters_args_container = {key: (value if type(value) in {str, int, float, bool} else str(value)) for
                                     key, value in
                                     vars(self.args).items() if key in MCLExperiment.parameters_names or
                                     key in MCLExperiment.preprocess_parameters_names}
        self.results_key = json.dumps(parameters_args_container)
        self.results_key = self.results_key.replace('.', '-')  # periods are reserved to groups in hdf5
        self.results_key = self.results_key.replace('/', '|')  # slashes are reserved to groups in hdf5
        self.results_key = self.results_key.replace(' ', '_')
        self.results_key = self.results_key.replace('{', '')
        self.results_key = self.results_key.replace('}', '')
        self.results_dir = self.save_dir / 'results'
        self.run_results_dir = None  # alternative to h5 saving
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.h5_results_file = None
        self.log = {'date': str(datetime.now())}
        self.som = None
        self.run_key = None
        self.clustering, self.clusters, self.matrix = (None,) * 3

    def __del__(self):
        if self.h5_results_file is not None:
            self.h5_results_file.close()  # ensure hdf5 file gets closed, to prevent corruption of its data
        # TODO CHECK: is this working during debugging?

    @staticmethod
    def name():
        return "MCLExperiment"

    @classmethod
    def modify_commandline_options(cls, parser):
        parser = super().modify_commandline_options(parser)
        parser.add_argument('--segmentation_experiment', type=str, required=True)
        parser.add_argument('--features_filename', type=str, default='filtered.h5',
                            help="Basename of files storing features")
        parser.add_argument('--outlier_removal', action='store_true')
        parser.add_argument('--isolation_contamination', default='auto',
                            help="Contamination parameter in the isolation  (auto or float)")
        parser.add_argument('--mcl_pruning_frequency', type=int, default=1,
                            help="Perform pruning every 'pruning_frequency' iterations.")
        parser.add_argument('--overwrite', action='store_true', help="Whether to discard old models and overwrite them")
        parser.add_argument('--gleason_file', type=Path,
                            default=r'/well/rittscher/projects/TCGA_prostate/TCGA/data/gdc_download_20190827_173135.969230/06efd272-a76f-4703-98b8-dfa751c0f019/nationwidechildrens.org_clinical_patient_prad.txt')
        return parser

    def read_data(self):
        # NB: one should be careful not to launch two processes that will write to the same results file, as multiple
        # programmes writing into the file at the same time will damage it irreparably. adding the options key to the
        # file adds a safeguard, but does not prevent damage in case multiple writers have the same options
        self.h5_results_file = h5py.File(self.save_dir / 'results' / f'results_{self.results_key}.h5', mode='w',
                                         libver='latest', swmr=True)
        h5_results_group = self.h5_results_file.create_group(self.results_key)  # store args
        # data clean-up; remove outliers
        features_file = self.args.data_dir / 'data' / 'features' / self.args.segmentation_experiment / self.args.features_filename
        try:
            features = pd.read_hdf(features_file.parent / 'single_key.h5', 'features')
        except FileNotFoundError:
            slide_ids = list(h5py.File(features_file, 'r').keys())  # read keys to access different stored frames
            feature_frames = []
            for slide_id in tqdm(slide_ids, desc=f"Reading features for {len(slide_ids)} slides ..."):
                feature_frame = pd.read_hdf(features_file, slide_id)
                slide_id = slide_id[:-1]  # BUG SOMEWHERE INCLUDED A PERIOD AT THE END OF KEYS -- remove
                # form multiindex with both bounding box and slide id information
                feature_frame.index = pd.MultiIndex.from_arrays(((slide_id,) * len(feature_frame), feature_frame.index),
                                                                names=('slide_id', 'bounding_box'))
                feature_frames.append(feature_frame)
            features = pd.concat(feature_frames)
            features.to_hdf(features_file.parent / 'single_key.h5', 'features')
        print(f"Features were read (size: {bytes2human(features.memory_usage().sum())})")
        print(f"Data shape: {features.shape}")
        # if __debug__:  # subsample features
        #     features = features.sample(n=1000)
        if self.args.isolation_contamination.isnumeric():
            self.args.isolation_contamination = float(self.args.isolation_contamination)
        # normalize features before outlier removal
        features = pd.DataFrame(data=StandardScaler().fit_transform(features), columns=features.columns,
                                index=features.index)
        if self.args.outlier_removal:
            # since the clustering is done on the codebooks, remembering where the inliers are is not strictly necessary
            # but we log it for reproducibility
            try:
                with open(self.save_dir / 'inliers_list.json', 'r') as inliers_list_file:
                    inliers = json.load(inliers_list_file)
            except FileNotFoundError:
                outlier_remover = IsolationForest(contamination=self.args.isolation_contamination)
                n_before = len(features)
                # below: need to cast to bool or a np boolean is returned + need to use list as tuple is considered a key by []
                inliers = list(bool(label != -1) for label in outlier_remover.fit_predict(features))
                with open(self.save_dir / 'inliers_list.json', 'w') as inliers_list_file:
                    json.dump(inliers, inliers_list_file)
            features = features.loc[inliers]
            self.inliers = inliers
            n_after = len(features)
            print(f"Removed {n_before - n_after} outliers through {str(outlier_remover)}")
            self.log.update({'outlier_removal': True, 'num_removed_outliers': n_before - n_after})
        else:
            self.log.update({'outlier_removal': False, 'num_removed_outliers': 0})
        h5_results_group.create_dataset('features', data=np.array(features))
        self.features = features

    def preprocess(self, parameters=None):
        if parameters is None:
            Parameters = namedtuple('Parameters', ['map_size', 'rough_epochs', 'finetune_epochs'])
            parameters = Parameters(
                map_size=read_parameter_values(self.args, 'map_size')[0],
                rough_epochs=read_parameter_values(self.args, 'rough_epochs')[0],
                finetune_epochs=read_parameter_values(self.args, 'finetune_epochs')[0]
            )
        # train som
        self.som = train_som(self.features, (parameters.map_size, parameters.map_size), self.save_dir,
                             parameters.rough_epochs, parameters.finetune_epochs)
        print(f"Codebook shape: {self.som.codebook.matrix.shape}")

    def run(self, parameters=None):
        if parameters is None:
            raise ValueError("parameters cannot be None")
        self.run_parameters = parameters
        self.run_key = self.format_parameters_key(self.run_parameters)
        self.run_key = self.run_key.replace('.', '-')  # periods are reserved to groups in hdf5
        self.run_key = self.run_key.replace('/', '|')  # slashes are reserved to paths in hdf5
        self.run_key = self.run_key.replace(' ', '_')
        self.run_key = self.run_key.replace('{', '')
        self.run_key = self.run_key.replace('}', '')
        self.run_results_dir = self.results_dir / self.format_parameters_key(self.run_parameters)
        self.run_results_dir.mkdir(exist_ok=True, parents=True)
        h5_results_group = self.h5_results_file[self.results_key]
        h5_iteration_group = h5_results_group.create_group(self.run_key)
        codebooks = self.som.codebook.matrix
        # simplicial set construction
        try:
            try:
                self.matrix = np.load(self.run_results_dir / 'simplicial_matrix.npy')
            except FileNotFoundError:
                self.matrix = h5_iteration_group['simplicial_matrix']
                np.save(self.run_results_dir / 'simplicial_matrix.npy', self.matrix.toarray())
        except KeyError:
            tqdm.write("Constructing a simplicial set on codebooks ...")
            with warnings.catch_warnings():  # ignore numba compile warnings
                warnings.simplefilter("ignore")
                self.matrix = umap.umap_.fuzzy_simplicial_set(X=codebooks, n_neighbors=parameters.num_neighbors,
                                                              random_state=np.random.RandomState(0), metric='euclidean')
                h5_iteration_group.create_dataset('simplicial_matrix', data=self.matrix.toarray())
                np.save(self.run_results_dir / 'simplicial_matrix.npy', self.matrix.toarray())
        # markov clustering
        try:
            try:
                self.clustering = jl.load(self.run_results_dir / 'clustering.joblib')
                with open(self.run_results_dir / 'clusters.json', 'r') as clusters_file:
                    self.clusters = json.load(clusters_file)
            except FileNotFoundError:
                self.clustering = h5_iteration_group['clustering']
                jl.dump(self.clustering, self.run_results_dir / 'clustering.joblib')
                self.clusters = [h5_iteration_group[f'cluster{i}'] for i in range(h5_iteration_group['num_clusters'])]
                with open(self.run_results_dir / 'clusters.json', 'w') as clusters_file:
                    json.dump(self.clusters, clusters_file)
            tqdm.write(
                f"Loaded markov clustering parameters for {parameters.num_neighbors} neighbors and inflation={parameters.inflation}")
        except KeyError:
            # RUN MARKOV CLUSTERING
            tqdm.write(
                f"Running markov clustering for {parameters.num_neighbors} neighbors and inflation={parameters.inflation}")
            if not isspmatrix_csr(self.matrix):
                self.matrix = csr_matrix(self.matrix)
            self.clustering = mc.run_mcl(self.matrix, inflation=parameters.inflation,
                                         pruning_frequency=self.args.mcl_pruning_frequency,
                                         verbose=True)  # NB mcl.prune breaks with scipy>0.13, see 04/10/2019 log
            h5_iteration_group.create_dataset('clustering', data=self.clustering.toarray())
            jl.dump(self.clustering, self.run_results_dir / 'clustering.joblib')
            self.clusters = mc.get_clusters(self.clustering)
            for i, cluster_nodes in enumerate(self.clusters):
                cluster_nodes = [int(n) for n in cluster_nodes]
                h5_iteration_group.create_dataset(f'cluster{i}', data=np.array(cluster_nodes))
            h5_iteration_group.create_dataset('num_clusters', data=len(self.clusters))
            with open(self.run_results_dir / 'clusters.json', 'w') as clusters_file:
                json.dump(self.clusters, clusters_file)
        tqdm.write(f"Number of clusters: {len(self.clusters)}")

    def evaluate(self):
        # assign membership to feature points
        num_clusters = len(self.clusters)
        try:
            som_cluster_membership = np.load(self.run_results_dir / 'som_cluster_membership.npy')
        except FileNotFoundError:
            som_cluster_assignment = np.vstack(
                [[(x, idx) for x in cluster] for idx, cluster in enumerate(self.clusters)])
            som_cluster_assignment = som_cluster_assignment[som_cluster_assignment[:, 0].argsort()]
            som_cluster_membership = som_cluster_assignment[:, 1]
            np.save(self.run_results_dir / 'som_cluster_membership.npy', som_cluster_membership)
        nearest_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.som.codebook.matrix)
        distance, indices = nearest_neighbours.kneighbors(self.features)  # find nearest SOM cluster for each data-point
        distance = distance.ravel()
        indices = indices.ravel()
        # assign each point to the markov cluster of his nearest codebook
        cluster_assignment = pd.Series(data=[som_cluster_membership[idx] for idx in indices], index=self.features.index)
        membership_histogram = np.histogram(cluster_assignment, bins=num_clusters, range=(0, num_clusters))[0]
        tqdm.write(f"Computing modularity for inflation={self.run_parameters.inflation} ...")
        try:
            try:
                with open(self.run_results_dir / 'evaluation_results.json', 'r') as evaluation_results_file:
                    results = json.load(evaluation_results_file)
                Q = results['modularity']
            except FileNotFoundError:
                Q = self.h5_results_file[self.results_key][self.run_key]['modularity']
        except KeyError:
            Q = mc.modularity(matrix=self.clustering,
                              clusters=self.clusters)  # this step is very long for low inflation
            self.h5_results_file[self.results_key][self.run_key].create_dataset('modularity', data=Q)
        tqdm.write(f"Modularity for inflation={self.run_parameters.inflation} is {Q}.")
        results = {
            'modularity': Q,
            'clusters': self.clusters,
            'num_clusters': num_clusters,
            'membership_histogram': list(int(n) for n in membership_histogram),
            'median_cluster_coverage': round((np.median(membership_histogram) / len(self.features)), 2),
            'max_cluster_coverage': round((membership_histogram.max() / len(self.features)), 2),
            'min_cluster_coverage': round((membership_histogram.min() / len(self.features)), 2),
            'calinski_harabasz_score': calinski_harabasz_score(self.features, cluster_assignment),
            'davies_bouldin_score': davies_bouldin_score(self.features, cluster_assignment),
            'silhouette_score': silhouette_score(self.features, cluster_assignment),
        }
        # extract umap embedding to create vizualization for markov clustering
        codebooks = self.som.codebook.matrix
        embedding_umap = umap.UMAP(n_neighbors=self.run_parameters.num_neighbors,
                                   min_dist=0.3,
                                   metric='euclidean').fit_transform(codebooks)
        # visualize markov clustering graph
        mc.draw_graph(self.matrix, self.clusters, pos=embedding_umap, node_size=50,
                      with_labels=False, edge_color="silver")
        viz_path = Path(self.run_results_dir, f'mcl_viz.png')
        plt.savefig(viz_path)
        plt.close()
        cluster_assignment = pd.Series(data=[som_cluster_membership[idx] for idx in indices], index=self.features.index)
        # histogram of cluster membership
        plt.hist(cluster_assignment)
        plt.title('cluster_assignment')
        plt.xlabel('cluster #')
        plt.ylabel('num glands')
        plt.close()
        cluster_assignment.to_csv(self.run_results_dir / f'cluster_assignment.csv')
        # calculate cluster histograms
        num_clusters = len(np.unique(cluster_assignment))
        histograms = dict()
        for slide_id in np.unique(cluster_assignment.index.get_level_values('slide_id')):
            assignments = cluster_assignment.loc[slide_id]
            histograms[slide_id] = np.histogram(assignments, bins=num_clusters, range=(0, num_clusters), density=True)[
                0]
        histograms = pd.DataFrame(histograms).T  # slide ids will be index values instead of columns
        # read gleason file
        gleason_table = pd.read_csv(self.args.gleason_file, delimiter='\t', skiprows=lambda x: x in [1, 2])
        # construct labels for the histogram data-points
        labels = []
        for slide_id, row in histograms.iterrows():
            subtable = gleason_table[gleason_table['bcr_patient_barcode'].str.startswith(slide_id[:12])]
            assert (len(subtable) == 1)
            if subtable['gleason_pattern_primary'].iloc[0] == 3 and subtable['gleason_pattern_secondary'].iloc[0] == 4:
                labels.append('3+4')
            elif subtable['gleason_pattern_primary'].iloc[0] == 4 and subtable['gleason_pattern_secondary'].iloc[
                0] == 3:
                labels.append('4+3')
            elif subtable['gleason_score'].iloc[0] == 6:
                labels.append('low')
            else:
                labels.append('high')
        # train to predict gleason from clusters
        labels_rfc = [{'low': 0, '3+4': 1, '4+3': 2, 'high': 3}[label] for label in labels]
        scores, rfc_importances = [], []
        for i in range(100):
            x_train, x_test, y_train, y_test = train_test_split(histograms, labels_rfc, train_size=0.5)
            rfc = RandomForestClassifier(n_estimators=100)
            rfc.fit(x_train, y_train)
            scores.append(rfc.score(x_test, y_test))
            rfc_importances.append(rfc.feature_importances_)
        average_score = np.mean(scores)
        feature_importances = list(float(f) for f in np.array(rfc_importances).mean(axis=0))
        # cluster in K means space to assess feature space separation
        selected = SelectPercentile(lambda X, y: RandomForestClassifier(n_estimators=100).fit(X, y).feature_importances_,
                                    percentile=15).fit_transform(histograms, labels_rfc)
        comparison_labels = KMeans(n_clusters=4).fit_predict(selected)
        assert set(labels_rfc) == set(comparison_labels)
        # gather examples from each cluster
        results.update({
            'rf_average_gleason_prediction_score': average_score,
            'feature_importances': feature_importances,
            'adjusted_mutual_information_score': adjusted_mutual_info_score(labels_rfc, comparison_labels),
            'adjusted_rand_score': adjusted_rand_score(labels_rfc, comparison_labels)
        })
        with open(self.run_results_dir / 'evaluation_results.json', 'w') as evaluation_results_file:
            json.dump(results, evaluation_results_file)
        # gather examples from each cluster
        membership_numbers = [np.sum(cluster_assignment == p) for p in np.unique(som_cluster_membership)]
        cluster_indices = list(
            p for p in np.unique(som_cluster_membership) if membership_numbers[p] > 10)  # exclude very small clusters
        cluster_centers = tuple(self.som.codebook.matrix[np.where(som_cluster_membership == i)[0]]
                                for i in cluster_indices)
        examples = get_cluster_examples(self.features, cluster_assignment,
                                        image_dir=self.args.data_dir,
                                        cluster_centers=cluster_centers,
                                        clusters=cluster_indices,
                                        n_examples=16)
        make_cluster_grids(examples, self.run_results_dir, self.name(), image_size=512)
        return results

    def save_results(self, results):
        formatted_results = {self.format_parameters_key(parameters): value for parameters, value in results.items()}
        with open(self.save_dir / f'{self.name().lower()}_results_{str(datetime.now())[:10]}.json',
                  'w') as results_file:
            json.dump(formatted_results, results_file)

    def select_best(self, results):
        # clear data
        self.som, self.matrix, self.clustering, self.clusters = (None,) * 4
        # select best based on modularity measure
        parameters_values, modularities = [], []
        for parameters, results_values in results.items():
            parameters_values.append(parameters)
            modularities.append(results_values['modularity'])
        return parameters_values[np.argmax(modularities)]

    def postprocess(self, best_parameters, best_result):
        # load simplicial matrix for use below
        best_result['matrix'] = np.load(
            self.results_dir / self.format_parameters_key(best_parameters) / 'simplicial_matrix.npy')
        self.matrix = best_result['matrix']
        # load som
        best_som_path = self.save_dir / f'model_size:{best_parameters.map_size}_re:{best_parameters.rough_epochs}_fte{best_parameters.finetune_epochs}.joblib'
        self.som = jl.load(best_som_path)
        self.h5_results_file[self.results_key].create_dataset('codebook_matrix',
                                                              data=np.array(self.som.codebook.matrix))
        final_results_dir = self.save_dir / 'final_result'
        final_results_dir.mkdir(exist_ok=True, parents=True)
        codebooks = self.som.codebook.matrix
        # extract umap embedding to create vizualization for markov clustering
        embedding_umap = umap.UMAP(n_neighbors=best_parameters.num_neighbors,
                                   min_dist=0.3,
                                   metric='euclidean').fit_transform(codebooks)
        som_cluster_assignment = np.vstack(
            [[(x, idx) for x in cluster] for idx, cluster in enumerate(best_result['clusters'])])
        som_cluster_assignment = som_cluster_assignment[som_cluster_assignment[:, 0].argsort()]
        som_cluster_membership = som_cluster_assignment[:, 1]
        np.save(final_results_dir / 'som_cluster_membership.npy', som_cluster_membership)
        h5_results_group = self.h5_results_file[self.results_key]
        h5_results_group.create_dataset('final_som_cluster_membership', data=som_cluster_membership)
        self.h5_results_file[self.results_key].create_dataset('final_num_clusters', data=np.max(som_cluster_membership))
        mc.draw_graph(best_result['matrix'], best_result['clusters'], pos=embedding_umap, node_size=50,
                      with_labels=False, edge_color="silver")
        viz_path = Path(self.results_dir, f'mcl_{self.format_parameters_key(best_parameters)}.png')
        plt.savefig(viz_path)
        plt.close()
        h5_results_group.create_dataset('mcl_viz', data=imageio.imread(viz_path))
        h5_results_group.create_dataset('embedding_map', data=embedding_umap)
        h5_results_group.create_dataset('final_simplicial_set_matrix', data=best_result['matrix'])
        for i, cluster_nodes in enumerate(best_result['clusters']):
            cluster_nodes = [int(n) for n in cluster_nodes]
            h5_results_group.create_dataset(f'final_cluster{i}', data=np.array(cluster_nodes))
        print(f"Final number of clusters: {np.max(som_cluster_membership)}")
        self.log['num_clusters'] = int(np.max(som_cluster_membership))
        save_log(self.log, self.args.data_dir)
        # assign each data-point to a code vector
        nearest_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.som.codebook.matrix)
        distance, indices = nearest_neighbours.kneighbors(self.features)  # find nearest SOM cluster for each data-point
        distance = distance.ravel()
        indices = indices.ravel()
        # assign each point to the markov cluster of his nearest codebook
        cluster_assignment = pd.Series(data=[som_cluster_membership[idx] for idx in indices], index=self.features.index)
        plt.hist(cluster_assignment)
        plt.title('cluster_assignment')
        plt.savefig(final_results_dir / f'cluster_assignment_{self.format_parameters_key(best_parameters)}.png')
        plt.close()
        cluster_assignment.to_csv(
            final_results_dir / f'cluster_assignment_{self.format_parameters_key(best_parameters)}.csv')
        # gather examples from each cluster
        membership_numbers = [np.sum(cluster_assignment == p) for p in np.unique(som_cluster_membership)]
        cluster_indices = list(
            p for p in np.unique(som_cluster_membership) if membership_numbers[p] > 10)  # exclude very small clusters
        cluster_centers = tuple(self.som.codebook.matrix[np.where(som_cluster_membership == i)[0]]
                                for i in cluster_indices)
        examples = get_cluster_examples(self.features, cluster_assignment,
                                        image_dir=self.args.data_dir,
                                        cluster_centers=cluster_centers)
        make_cluster_grids(examples, final_results_dir, self.name(), image_size=512)
        print("Done!")


def save_log(log, data_dir):
    (data_dir / 'data' / 'logs').mkdir(exist_ok=True, parents=True)
    with (data_dir / 'data' / 'logs' / f'tcga_analysis_{log["date"]}.json').open('w') as log_file:
        json.dump(log, log_file)


def train_som(data, mapsize, save_dir, rough_epochs=30, finetune_epochs=100, overwrite=False):
    # sequential version of above
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f'model_size:{mapsize[0]}_re:{rough_epochs}_fte{finetune_epochs}.joblib'
    if overwrite or not save_path.exists():  # SOM
        sm = SOMFactory.build(data,
                              mapsize=mapsize,
                              normalization=None,
                              initialization='pca',
                              lattice="rect")
        sm.train(n_job=1, verbose='info', train_rough_len=rough_epochs, train_finetune_len=finetune_epochs)
        jl.dump(sm, save_path)
    else:
        sm = jl.load(save_path)
    return sm

# def sub_train_som(data, mapsize):
#     sm = SOMFactory.build(data,
#                           mapsize=mapsize,
#                           normalization=None,
#                           initialization='random',
#                           lattice="rect")
#     sm.train(n_job=1, verbose='info', train_rough_len=30, train_finetune_len=100)
#     return sm
#
#
# def train_som_para(data, mapsize, save_dir, train_rough_len=30, train_finetune_len=100, overwrite=False):
#     save_dir = Path(save_dir)
#     save_dir.mkdir(exist_ok=True, parents=True)
#     save_path = save_dir / f'model_size:{mapsize}_.joblib'
#     if overwrite or not save_path.exists():
#         # SOM
#         sm_set = jl.Parallel(n_jobs=4)(jl.delayed(sub_train_som)(data, mapsize) for _ in range(50))
#         sm_idx = np.argmin([sm.calculate_topographic_error() for sm in sm_set])
#         sm = sm_set[sm_idx]
#         jl.dump(sm, save_path)
#
#     else:
#         sm = jl.load(save_path)
#     return sm
