from pathlib import Path
import json
from datetime import datetime
from collections import namedtuple
import warnings
from copy import copy
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import joblib as jl
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sompy.sompy import SOMFactory
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, \
    adjusted_mutual_info_score, adjusted_rand_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py
import umap
import markov_clustering as mc
from base.utils.utils import bytes2human
from quant.experiment import BaseExperiment  # should inherit from clustering instead ?
from quant.utils import read_parameter_values
from quant.viz import get_cluster_examples, make_cluster_grids
import logging  # disable annoying font messages


logging.getLogger('matplotlib.font_manager').disabled = True


if float(pd.__version__[:3]) >= 1.0:
    raise EnvironmentError("joblib doesn't work with pandas >= 1.0 because FrozenNDArray was removed")
#https://github.com/pandas-dev/pandas/pull/29335/files/67a7241447abbcc76699b05be523e93143857f25


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
        self.distributions = None

    def __del__(self):
        if self.h5_results_file is not None:
            self.h5_results_file.close()  # ensure hdf5 file gets closed, to prevent corruption of its data

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
        parser.add_argument('--min_cluster_size', type=int, default=10, help="Clusters with less examples than this number will be discarded")
        parser.add_argument('--overwrite', action='store_true', help="Whether to discard old models and overwrite them")
        parser.add_argument('--gleason_file', type=Path,
                            default=r'/well/rittscher/projects/TCGA_prostate/TCGA/data/gdc_download_20190827_173135.969230/06efd272-a76f-4703-98b8-dfa751c0f019/nationwidechildrens.org_clinical_patient_prad.txt')
        parser.add_argument('--train_fraction', type=float, default=0.7, help="fraction of dataset used to train rf for gleason prediction")
        parser.add_argument('--full_histograms', action='store_true', help="whether to normalize cluster histograms before grade classification")
        parser.add_argument('--labels_file', type=Path, default=None)
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
            features = pd.read_hdf(features_file.parent / f'single_key_{self.args.features_filename[:-3]}.h5', 'features')
        except FileNotFoundError:
            slide_ids = list(h5py.File(features_file, 'r').keys())  # read keys to access different stored frames
            feature_frames, unreadable_slides = [], []
            for slide_id in tqdm(slide_ids, desc=f"Reading features for {len(slide_ids)} slides ..."):
                try:
                    feature_frame = pd.read_hdf(features_file, slide_id)
                except ValueError as err:
                    print(err)
                    unreadable_slides.append(slide_id)
                    continue
                slide_id = slide_id[:-1]  # BUG SOMEWHERE INCLUDED A PERIOD AT THE END OF KEYS -- remove
                # form multiindex with both bounding box and slide id information
                feature_frame.index = pd.MultiIndex.from_arrays(((slide_id,) * len(feature_frame), feature_frame.index),
                                                                names=('slide_id', 'bounding_box'))
                feature_frames.append(feature_frame)
            print(f"Features for {len(unreadable_slides)} slides could not be read:\n{unreadable_slides}")
            features = pd.concat(feature_frames)
            features.to_hdf(features_file.parent/f'single_key_{self.args.features_filename[:-3]}.h5', 'features')
        print(f"Features were read (size: {bytes2human(features.memory_usage().sum())})")
        print(f"Data shape: {features.shape}")
        # if self.args.labels_file:
        #     original_features_len = len(features)
        #     box_ids = features.index.get_level_values('bounding_box').tolist()
        #     to_drop = []
        #     labels = pd.read_csv(self.args.labels_file)
        #     benign_labels = labels[labels['positive_probability'] < 0.5]
        #     for _, row in benign_labels.iterrows():
        #         try:
        #             to_drop.append(box_ids.index(row['box_id']))  # FIXME superslow
        #         except ValueError:
        #             pass
        #     features.drop(to_drop, inplace=True)
        #     print(f"Removed {len(features) - original_features_len} benign glands")
        if self.args.debug:  # subsample features
            features = features.sample(n=1000)
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
        # read distance histograms
        distances_dir = self.args.data_dir / 'data' / 'features' / self.args.segmentation_experiment / 'relational'
        with open(distances_dir/'distance_distributions.json', 'r') as distributions_file:
            self.distributions = pd.read_json(distributions_file)
        self.distributions.index.rename('slide_id', inplace=True)
        self.distributions.drop(
            set(self.distributions.index.unique(level='slide_id')) - set(self.features.index.unique(level='slide_id')),
            inplace=True
        )  # drop distributions for slides that don't have features
        print(f"Distributions were loaded: shape = {self.distributions.shape}")
        assert len(self.distributions) == len(self.features.index.unique(level='slide_id')), f"Length of distributions differs from length of features {len(self.distributions)} ≠ {len(self.features.index.unique(level='slide_id'))}"

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

    def embed_in_fuzzy_sets(self, parameters, codebooks, h5_iteration_group):
        tqdm.write("Computing codebooks embedding ...")
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

    def markov_cluster(self, parameters, h5_iteration_group):
        tqdm.write("Performing Markov Clustering on the codebooks embedding ...")
        try:
            try:
                self.clustering = jl.load(self.run_results_dir / 'clustering.joblib')
                with open(self.run_results_dir / 'clusters.json', 'r') as clusters_file:
                    self.clusters = json.load(clusters_file)
                if sum(len(c) for c in self.clusters) != self.matrix.shape[0]:
                    raise FileNotFoundError("Clusters overlap")
            except FileNotFoundError:
                self.clustering = h5_iteration_group['clustering']
                jl.dump(self.clustering, self.run_results_dir / 'clustering.joblib')
                self.clusters = [h5_iteration_group[f'cluster{i}'] for i in range(h5_iteration_group['num_clusters'])]
                if sum(len(c) for c in self.clusters) != self.matrix.shape[0]:
                    raise KeyError("Clusters overlap")
                with open(self.run_results_dir / 'clusters.json', 'w') as clusters_file:
                    json.dump(self.clusters, clusters_file)
            tqdm.write(
                f"Loaded markov clustering parameters for {parameters.num_neighbors} neighbors and inflation={parameters.inflation}")
        except KeyError:
            # RUN MARKOV CLUSTERING
            tqdm.write(f"Running markov clustering for {parameters.num_neighbors} neighbors and inflation={parameters.inflation}")
            if not isspmatrix_csr(self.matrix):
                self.matrix = csr_matrix(self.matrix)
            self.clustering = mc.run_mcl(self.matrix, inflation=parameters.inflation,
                                         pruning_frequency=self.args.mcl_pruning_frequency,
                                         verbose=True)  # NB mcl.prune breaks with scipy>0.13, see 04/10/2019 log
            h5_iteration_group.create_dataset('clustering', data=self.clustering.toarray())
            jl.dump(self.clustering, self.run_results_dir / 'clustering.joblib')
            self.clusters = mc.get_clusters(self.clustering)
            self.clusters = self.trim_double_cluster_assignments(self.clusters, h5_iteration_group)
            assigned_nodes = set()  # get_clusters() can assign a node to two clusters
            h5_iteration_group.create_dataset('num_clusters', data=len(self.clusters))
            with open(self.run_results_dir / 'clusters.json', 'w') as clusters_file:
                json.dump(self.clusters, clusters_file)

    def trim_double_cluster_assignments(self, clusters, h5_iteration_group=None):
        r"""mc.get_clusters() assigns some codebooks to two clusters. Trim doubles away with this function"""
        clusters = [list(c) for c in clusters]  # so that repeated elements can be removed below
        assigned_nodes = set()  # get_clusters() can assign a node to two clusters
        for i, cluster_nodes in enumerate(copy(clusters)):  # FIXME still crashing because of extra index?
            for ci in cluster_nodes:
                if ci in assigned_nodes:
                    clusters[i].remove(ci)
                assigned_nodes.add(ci)
            cluster_nodes = [int(n) for n in clusters[i]]
            if h5_iteration_group is not None:
                h5_iteration_group.create_dataset(f'cluster{i}', data=np.array(cluster_nodes))
        return clusters

    def set_run_properites(self, parameters):
        self.run_parameters = parameters
        self.run_key = self.format_parameters_key(self.run_parameters)
        self.run_key = self.run_key.replace('.', '-')  # periods are reserved to groups in hdf5
        self.run_key = self.run_key.replace('/', '|')  # slashes are reserved to paths in hdf5
        self.run_key = self.run_key.replace(' ', '_')
        self.run_key = self.run_key.replace('{', '')
        self.run_key = self.run_key.replace('}', '')
        self.run_results_dir = self.results_dir / self.format_parameters_key(self.run_parameters)
        self.run_results_dir.mkdir(exist_ok=True, parents=True)

    def run(self, parameters=None):
        self.set_run_properites(parameters)
        if parameters is None:
            raise ValueError("parameters cannot be None")
        h5_results_group = self.h5_results_file[self.results_key]
        h5_iteration_group = h5_results_group.create_group(self.run_key)
        codebooks = self.som.codebook.matrix
        # simplicial set construction
        self.embed_in_fuzzy_sets(parameters, codebooks, h5_iteration_group)
        # markov clustering
        self.markov_cluster(parameters, h5_iteration_group)
        tqdm.write(f"Number of clusters: {len(self.clusters)}")

    def assign_membership(self, save_dir, parameters):
        tqdm.write("Assigning cluster membership to som codebooks and datapoints ...")
        if self.clusters is None:
            with open(self.results_dir/self.format_parameters_key(parameters)/ 'clusters.json', 'r') as clusters_file:
                self.clusters = json.load(clusters_file)
        num_clusters = len(self.clusters)
        try:
            som_cluster_membership = np.load(save_dir / 'som_cluster_membership.npy')
            if len(som_cluster_membership) != self.matrix.shape[0]:
                raise FileNotFoundError("Old best result -- must update assignment")
        except FileNotFoundError:
            som_cluster_assignment = np.vstack(
                [[(x, idx) for x in cluster] for idx, cluster in enumerate(self.clusters)])
            som_cluster_assignment = som_cluster_assignment[som_cluster_assignment[:, 0].argsort()]
            som_cluster_membership = som_cluster_assignment[:, 1]
            np.save(save_dir / 'som_cluster_membership.npy', som_cluster_membership)
        nearest_neighbours = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.som.codebook.matrix)
        distance, indices = nearest_neighbours.kneighbors(self.features)  # find nearest SOM codebook for each data-point
        distance = distance.ravel()
        indices = indices.ravel()
        # assign each point to the markov cluster of its nearest codebook
        cluster_assignment = pd.Series(data=[som_cluster_membership[idx] for idx in indices], index=self.features.index)
        membership_histogram = np.histogram(cluster_assignment, bins=num_clusters, range=(0, num_clusters))[0]
        # histogram of cluster membership
        fig = plt.figure()
        plt.hist(cluster_assignment)
        plt.title('cluster_assignment')
        plt.xlabel('cluster #')
        plt.ylabel('num glands')
        plt.close()
        fig.savefig(save_dir / f'cluster_assignment_{self.format_parameters_key(parameters)}.png')
        cluster_assignment.to_csv(save_dir / f'cluster_assignment_{self.format_parameters_key(parameters)}.csv')
        return cluster_assignment, membership_histogram, som_cluster_membership, num_clusters

    def compute_modularity(self, save_dir):
        try:
            try:
                with open(save_dir / 'evaluation_results.json', 'r') as evaluation_results_file:
                    results = json.load(evaluation_results_file)
                Q = results['modularity']
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                Q = self.h5_results_file[self.results_key][self.run_key]['modularity']
        except KeyError:
            Q = mc.modularity(matrix=self.clustering,
                              clusters=self.clusters)  # this step is very long for low inflation
            self.h5_results_file[self.results_key][self.run_key].create_dataset('modularity', data=Q)
        return Q

    def visualize_markov_clustering(self, save_path, parameters):
        # extract umap embedding to create vizualization for markov clustering
        viz_path = Path(save_path, f'mcl_{self.format_parameters_key(parameters)}.png')
        if not viz_path.exists():
            codebooks = self.som.codebook.matrix
            try:
                embedding_umap = np.load(save_path/'embedding_umap.npy')
            except FileNotFoundError:
                embedding_umap = umap.UMAP(n_neighbors=self.run_parameters.num_neighbors,
                                           min_dist=0.3,
                                           metric='euclidean').fit_transform(codebooks)
                np.save(save_path/'embedding_umap.npy', np.array(embedding_umap))
            # visualize markov clustering graph
            fig = plt.figure()
            mc.draw_graph(self.matrix, self.clusters, pos=embedding_umap, node_size=50,
                          with_labels=False, edge_color="silver")
            fig.savefig(viz_path)
            plt.close()

    def compute_per_slide_cluster_histograms(self, cluster_assignment, num_clusters, save_dir):
        # calculate cluster histograms per slide
        try:
            histograms = pd.read_csv(save_dir / 'histograms.csv', index_col=0)
        except FileNotFoundError:
            histograms = dict()
            for slide_id in np.unique(cluster_assignment.index.get_level_values('slide_id')):
                assignments = cluster_assignment.loc[slide_id]
                histograms[slide_id] = np.histogram(assignments, bins=num_clusters, range=(0, num_clusters),
                                                    density=not self.args.full_histograms)[0]
            histograms = pd.DataFrame(histograms).T  # slide ids will be index values instead of columns
            histograms.to_csv(save_dir / 'histograms.csv')
        return histograms

    def make_prototype_features(self, cluster_assignment):
        r"""Get representative features for every cluster"""
        cluster_assignment.name = 'cluster'
        clustered_features = pd.concat((self.features, cluster_assignment), axis=1)
        clustered_features.set_index('cluster', append=True, inplace=True)
        median_features = clustered_features.groupby(by=['slide_id', 'cluster']).median()
        # create data-frames with one row per slide id and one row per cluster
        clusters = np.unique(cluster_assignment)
        median_features = pd.DataFrame(PCA(n_components=20).fit_transform(median_features), index=median_features.index)
        median_features.sort_index(inplace=True)
        for slide_id, cluster in pd.MultiIndex.from_product((median_features.index.levels[0], clusters)):
            if (slide_id, cluster) not in median_features.index:
                median_features.append(pd.DataFrame(data=[[0]*20], index=[np.array([slide_id]), np.array([cluster])]),
                                       ignore_index=False)
        median_features.sort_index(inplace=True)
        median_features = median_features.unstack(level='cluster')
        return median_features

    def assess_gleason_predictiveness(self, slides_histograms, cluster_assignment, save_dir):
        tqdm.write("Assessing correlation between gleason score and clusters ...")
        # read gleason file
        gleason_table = pd.read_csv(self.args.gleason_file, delimiter='\t', skiprows=lambda x: x in [1, 2])
        # construct labels for the histogram data-points
        labels = []
        for slide_id, row in slides_histograms.iterrows():
            subtable = gleason_table[gleason_table['bcr_patient_barcode'].str.startswith(slide_id[:12])]
            assert (len(subtable) == 1)
            if subtable['gleason_pattern_primary'].iloc[0] + subtable['gleason_pattern_secondary'].iloc[0] == 7:
                labels.append('7')
            # if subtable['gleason_pattern_primary'].iloc[0] == 3 and subtable['gleason_pattern_secondary'].iloc[0] == 4:
            #     labels.append('3+4')
            # elif subtable['gleason_pattern_primary'].iloc[0] == 4 and subtable['gleason_pattern_secondary'].iloc[
            #     0] == 3:
            #     labels.append('4+3')
            elif subtable['gleason_score'].iloc[0] == 6:
                labels.append('low')
            else:
                labels.append('high')
        prototype_features = self.make_prototype_features(cluster_assignment)
        slide_features = pd.concat((slides_histograms, prototype_features, self.distributions), axis=1)  # TODO test
        # train to predict gleason from clusters
        # labels_rfc = [{'low': 0, '3+4': 1, '4+3': 2, 'high': 3}[label] for label in labels]
        labels_rfc = np.array([{'low': 0, '7': 1, 'high': 2}[label] for label in labels])
        # save dim reduction of histograms:
        dim_reduced = PCA(n_components=2).fit_transform(slides_histograms)
        figure = plt.figure()
        plt.scatter(dim_reduced[labels_rfc == 0][:, 0], dim_reduced[labels_rfc == 0][:, 1], c='r')
        plt.scatter(dim_reduced[labels_rfc == 1][:, 0], dim_reduced[labels_rfc == 1][:, 1], c='b')
        plt.scatter(dim_reduced[labels_rfc == 2][:, 0], dim_reduced[labels_rfc == 2][:, 1], c='g')
        plt.title('PCA of histograms')
        plt.xlabel('cluster #')
        plt.ylabel('num glands')
        plt.close()
        figure.savefig(save_dir/'histograms_pca.png')
        scores, rfc_importances, confusion_matrices = [], [], []
        try:
            average_score = np.load(save_dir/'average_score.npy')
            average_confusion_matrix = np.load(save_dir/'average_confusion_matrix.npy')
            feature_importances = np.load(save_dir/'feature_importances.npy')
            comparison_labels = np.load(save_dir/'comparison_labels.npy')
            with open(save_dir/'misclassified.json', 'r') as misclassified_file:
                misclassified = json.load(misclassified_file)
        except FileNotFoundError:
            # default is -1 for predictions and -2 for ground truth, so train entries in each column arent't counted as equal
            predictions, ground_truth = np.ones((len(slides_histograms), 100)) * -1, np.ones((len(slides_histograms), 100)) * -2
            for i in range(100):
                x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(slide_features, labels_rfc,
                                                                                         np.arange(len(slide_features)),
                                                                                         train_size=self.args.train_fraction)
                rfc = XGBClassifier(n_estimators=1000, max_depth=40, subsample=0.9)
                rfc.fit(x_train, y_train)
                y_pred = rfc.predict(x_test)
                predictions[idx_test, i] = np.array(y_pred)
                ground_truth[idx_test, i] = np.array(y_test)
                confusion_matrices.append(confusion_matrix(y_test, y_pred))  # compute the confusion matrix
                scores.append(rfc.score(x_test, y_test))
                rfc_importances.append(rfc.feature_importances_)
            average_score = np.mean(scores)
            average_confusion_matrix = np.mean(np.stack(confusion_matrices, axis=0), axis=0)
            print(f"Gleason classification score is {average_score}")
            print(f"Confusion matrix is:")
            print(average_confusion_matrix.tolist())  # TODO change to prettyprint once scripts have run
            feature_importances = np.array(list(float(f) for f in np.array(rfc_importances).mean(axis=0)))
            prediction_rates = ((predictions == ground_truth).sum(axis=-1) /
                      np.array([pred[pred != -1].sum() for pred in predictions]))
            misclassified = [slide_id for slide_id, accuracy in zip(tuple(slides_histograms.index), prediction_rates) if accuracy > 0.5]
            # cluster in K means space to assess feature space separation
            comparison_labels, n = set(), 3
            selected = SelectPercentile(
                lambda X, y: XGBClassifier(n_estimators=100).fit(X, y).feature_importances_,
                percentile=5*n).fit_transform(slides_histograms, labels_rfc)
            comparison_labels = KMeans(n_clusters=3).fit_predict(selected)
            assert set(labels_rfc) == set(comparison_labels)
            np.save(save_dir/'average_score.npy', average_score)
            np.save(save_dir/'average_confusion_matrix.npy', average_confusion_matrix)
            np.save(save_dir/'feature_importances', feature_importances)
            np.save(save_dir/'comparison_labels', comparison_labels)
            np.save(save_dir/'labels_rfc.npy', labels_rfc)
            with open(save_dir/'misclassified.json', 'w') as misclassified_file:
                json.dump(misclassified, misclassified_file)
        return average_score, average_confusion_matrix, feature_importances, comparison_labels, labels_rfc, misclassified

    def save_cluster_examples(self, cluster_assignment, som_cluster_membership, save_dir):
        # gather examples from each cluster (exclude very small clusters)
        membership_numbers = [np.sum(cluster_assignment == p) for p in np.unique(som_cluster_membership)]
        cluster_indices = list(p for p in np.unique(som_cluster_membership) if membership_numbers[p] > 10)
        np.save(save_dir / 'som_cluster_membership.npy', som_cluster_membership)
        cluster_centers = tuple(self.som.codebook.matrix[np.where(som_cluster_membership == i)[0]]
                                for i in cluster_indices)
        with open(save_dir/'cluster_centers.joblib', 'wb') as cluster_centers_file:
            jl.dump(cluster_centers, cluster_centers_file)  # TODO test
        examples = get_cluster_examples(self.features, cluster_assignment,
                                        image_dir=self.args.data_dir,
                                        cluster_centers=cluster_centers,
                                        clusters=cluster_indices,
                                        n_examples=16)
        make_cluster_grids(examples, save_dir, self.name(), image_size=512)

    def evaluate(self):
        tqdm.write("Evaluating run results ...")
        # assign membership to feature points
        cluster_assignment, membership_histogram, som_cluster_membership, num_clusters = \
            self.assign_membership(self.run_results_dir, self.run_parameters)
        tqdm.write(f"Computing modularity for inflation={self.run_parameters.inflation} ...")
        Q = self.compute_modularity(self.run_results_dir)
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
        # save run markov clustering viz
        self.visualize_markov_clustering(self.run_results_dir, self.run_parameters)
        # computer cluster membership histograms for each slide
        slide_histograms = self.compute_per_slide_cluster_histograms(cluster_assignment, num_clusters, self.run_results_dir)
        average_score, average_confusion_matrix, feature_importances, comparison_labels, labels_rfc, misclassified = \
            self.assess_gleason_predictiveness(slide_histograms, cluster_assignment, self.run_results_dir)
        # gather examples from each cluster
        results.update({
            'rf_average_gleason_prediction_score': float(average_score),
            'feature_importances': feature_importances.tolist(),
            'adjusted_mutual_information_score': float(adjusted_mutual_info_score(labels_rfc, comparison_labels)),
            'adjusted_rand_score': float(adjusted_rand_score(labels_rfc, comparison_labels)),
            'average_confusion_matrix': average_confusion_matrix.tolist(),
            'misclassified_slides': list(misclassified)
        })
        with open(self.run_results_dir / 'evaluation_results.json', 'w') as evaluation_results_file:
            json.dump(results, evaluation_results_file)
        self.save_cluster_examples(cluster_assignment, som_cluster_membership, self.run_results_dir)
        return results

    def save_results(self, results):
        formatted_results = {self.format_parameters_key(parameters): value for parameters, value in results.items()}
        with open(self.save_dir / f'{self.name().lower()}_results_{str(datetime.now())[:10]}.json', 'w') as results_file:
            json.dump(formatted_results, results_file)

    def select_best(self, results):
        # clear data
        self.som, self.matrix, self.clustering, self.clusters = (None,) * 4
        # select best based on modularity measure
        parameters_values, rf_scores = [], []
        for parameters, results_values in results.items():
            parameters_values.append(parameters)
            rf_scores.append(results_values['rf_average_gleason_prediction_score'])
        return parameters_values[np.argmax(rf_scores)]

    def postprocess(self, best_parameters, best_result):
        tqdm.write("Evaluating best results ...")
        self.set_run_properites(best_parameters)
        self.clusters = best_result['clusters']
        # load simplicial matrix for use below
        best_result['matrix'] = np.load(
            self.results_dir / self.format_parameters_key(best_parameters) / 'simplicial_matrix.npy')
        self.matrix = best_result['matrix']
        # load markov clustering on graph
        self.clustering = jl.load(self.results_dir / self.format_parameters_key(best_parameters) / 'clustering.joblib')
        # load som
        best_som_path = self.save_dir / f'model_size:{best_parameters.map_size}_re:{best_parameters.rough_epochs}_fte{best_parameters.finetune_epochs}.joblib'
        self.som = jl.load(best_som_path)
        self.h5_results_file[self.results_key].create_dataset('codebook_matrix',
                                                              data=np.array(self.som.codebook.matrix))
        final_results_dir = self.save_dir / 'final_result'
        final_results_dir.mkdir(exist_ok=True, parents=True)
        cluster_assignment, membership_histogram, som_cluster_membership, num_clusters = \
            self.assign_membership(final_results_dir, best_parameters)
        Q = self.compute_modularity(final_results_dir)
        tqdm.write(f"Modularity for inflation={best_parameters.inflation} is {Q}.")
        h5_results_group = self.h5_results_file[self.results_key]
        h5_results_group.create_dataset('final_som_cluster_membership', data=som_cluster_membership)
        # save best markov clustering viz
        self.visualize_markov_clustering(final_results_dir, best_parameters)
        self.h5_results_file[self.results_key].create_dataset('final_num_clusters', data=np.max(som_cluster_membership))
        h5_results_group.create_dataset('final_simplicial_set_matrix', data=best_result['matrix'])
        for i, cluster_nodes in enumerate(best_result['clusters']):
            cluster_nodes = [int(n) for n in cluster_nodes]
            h5_results_group.create_dataset(f'final_cluster{i}', data=np.array(cluster_nodes))
        print(f"Final number of clusters: {np.max(som_cluster_membership)}")
        self.log['num_clusters'] = int(np.max(som_cluster_membership))
        save_log(self.log, self.args.data_dir)
        # gather examples from each cluster
        self.save_cluster_examples(cluster_assignment, som_cluster_membership, final_results_dir)
        print("Done!")

    def apply(self, parameters):  # apply clustering to dataset
        self.set_run_properites(parameters)
        dataset_savedir = self.save_dir/'applied_results'/(self.args.features_filename[:-3] + '_' + \
                          self.format_parameters_key(parameters))
        dataset_savedir.mkdir(exist_ok=True, parents=True)
        self.matrix = np.load(self.run_results_dir / 'simplicial_matrix.npy')
        with open(self.run_results_dir / 'clusters.json', 'r') as clusters_file:
            self.clusters = json.load(clusters_file)
        self.clusters = self.trim_double_cluster_assignments(self.clusters)
        # load som
        best_som_path = self.save_dir / f'model_size:{parameters.map_size}_re:{parameters.rough_epochs}_fte{parameters.finetune_epochs}.joblib'
        self.som = jl.load(best_som_path)
        cluster_assignment, membership_histogram, som_cluster_membership, num_clusters = \
            self.assign_membership(dataset_savedir, parameters)
        # assign each point to the markov cluster of his nearest codebook
        results = {
            'num_clusters': num_clusters,
            'membership_histogram': list(int(n) for n in membership_histogram),
            'median_cluster_coverage': round((np.median(membership_histogram) / len(self.features)), 2),
            'max_cluster_coverage': round((membership_histogram.max() / len(self.features)), 2),
            'min_cluster_coverage': round((membership_histogram.min() / len(self.features)), 2),
            'calinski_harabasz_score': calinski_harabasz_score(self.features, cluster_assignment),
            'davies_bouldin_score': davies_bouldin_score(self.features, cluster_assignment),
            'silhouette_score': silhouette_score(self.features, cluster_assignment)
        }
        slide_histograms = self.compute_per_slide_cluster_histograms(cluster_assignment, num_clusters, dataset_savedir)
        average_score, average_confusion_matrix, feature_importances, comparison_labels, labels_rfc, misclassified = \
            self.assess_gleason_predictiveness(slide_histograms, cluster_assignment, dataset_savedir)
        # gather examples from each cluster
        results.update({
            'rf_average_gleason_prediction_score': float(average_score),
            'feature_importances': feature_importances.tolist(),
            'adjusted_mutual_information_score': float(adjusted_mutual_info_score(labels_rfc, comparison_labels)),
            'adjusted_rand_score': float(adjusted_rand_score(labels_rfc, comparison_labels)),
            'misclassified_slides': list(misclassified)
        })
        with open(dataset_savedir/'evaluation_results.json', 'w') as evaluation_results_file:
            json.dump(results, evaluation_results_file)
        # gather examples from each cluster
        self.save_cluster_examples(cluster_assignment, som_cluster_membership, dataset_savedir)
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
