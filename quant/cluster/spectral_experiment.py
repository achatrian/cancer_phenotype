from pathlib import Path
from datetime import datetime
import json
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.cluster import SpectralClustering, KMeans # spectral clustering implementation in scikit-learn uses K-means for initialisation
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib as jl
from quant.experiment import BaseExperiment  # should inherit from clustering instead ?
from quant.utils.consensus_clustering import ConsensusCluster
from base.utils.utils import bytes2human
import logging
logging.getLogger('matplotlib.font_manager').disabled = True


class SpectralExperiment(BaseExperiment):
    r"""This class works with any clustering algorithm class that has
    a n_clusters parameter and a fit_predict method"""
    parameters_names = ('num_clusters',)
    parameters_defaults = ('5',)
    parameters_descriptions = ("Num of clusters",)

    def __init__(self, args):
        super().__init__(args)
        self.features, self.inliers = (None,) * 2
        self.save_dir = self.args.data_dir / 'data' / 'experiments' / f'{self.name().split("Experiment")[0].lower()}'
        self.save_dir.mkdir(exist_ok=True, parents=True)
        parameters_args_container = {key: (value if type(value) in {str, int, float, bool} else str(value)) for
                                     key, value in
                                     vars(self.args).items() if key in SpectralExperiment.parameters_names or
                                     key in SpectralExperiment.preprocess_parameters_names}
        self.results_key = json.dumps(parameters_args_container)
        self.results_dir = self.save_dir/'results'
        self.run_dir = None
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log = {'date': str(datetime.now())}
        self.run_key = None
        self.decomposed, self.reconstruction_error, self.components, self.run_iters = (None,) * 4
        self.labels = None

    @staticmethod
    def name():
        return "SpectralExperiment"

    @classmethod
    def modify_commandline_options(cls, parser):
        parser = super().modify_commandline_options(parser)
        parser.add_argument('--segmentation_experiment', type=str, required=True)
        parser.add_argument('--features_filename', type=str, default='filtered.h5',
                            help="Basename of files storing features")
        parser.add_argument('--low_num_clusters')
        parser.add_argument('--outlier_removal', action='store_true')
        parser.add_argument('--isolation_contamination', default='auto',
                            help="Contamination parameter in the isolation  (auto or float)")
        parser.add_argument('--clustering_type', choices=['kmeans', 'spectral'])
        parser.add_argument('--subsample_percentage', type=float, default=1.0, help="Subsample the whole dataset to allow for ")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--small_cluster_threshold', type=int, default=10)
        return parser

    def read_data(self):
        self.results_file = h5py.File(self.save_dir / 'results.h5', mode='w')
        results_group = self.results_file.create_group(self.results_key)  # store args
        # data clean-up; remove outliers
        features_file = self.args.data_dir/'data'/'features'/self.args.segmentation_experiment/self.args.features_filename
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
        if self.args.outlier_removal:
            outlier_remover = IsolationForest(contamination=self.args.isolation_contamination)
            n_before = len(features)
            # below: need to cast to bool or a np boolean is returned + need to use list as tuple is considered a key by []
            inliers = list(bool(label != -1) for label in outlier_remover.fit_predict(features))
            features = features.loc[inliers]
            self.inliers = inliers
            n_after = len(features)
            print(f"Removed {n_before - n_after} outliers through {str(outlier_remover)}")
            self.log.update({'outlier_removal': True, 'num_removed_outliers': n_before - n_after})
        else:
            self.log.update({'outlier_removal': False, 'num_removed_outliers': 0})
        results_group.create_dataset('features', data=np.array(features))  # TODO - test
        self.features = features

    def preprocess(self, parameters=None):
        pass

    def run(self, parameters=None):
        if parameters is None:
            raise ValueError("parameters cannot be None")
        tqdm.write(f"Cluster into {parameters.num_clusters} clusters ...")
        self.run_parameters = parameters
        self.run_dir = self.results_dir/self.format_parameters_key(parameters)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.args.subsample_percentage != 1.0 and len(self.features) > 10000:  # sample a subset of the whole dataset
            try:  # save a list of sampled datapoints
                with open(self.save_dir / f'subsample_indices.json', 'r') as subsample_indices_file:
                    sample_indices = json.load(subsample_indices_file)
            except FileNotFoundError:
                sample_indices = np.random.randint(0, len(self.features),
                                                   size=round(len(self.features)*self.args.subsample_percentage)).tolist()
                with open(self.save_dir/f'subsample_indices.json', 'w') as subsample_indices_file:
                    json.dump(sample_indices, subsample_indices_file)
            self.features = self.features.iloc[sample_indices]
        try:
            self.spectral_clustering = jl.load(self.run_dir/f'model.joblib')
        except FileNotFoundError:
            self.spectral_clustering = SpectralClustering(n_clusters=parameters.num_clusters, n_jobs=self.args.num_workers)  # n_jobs=-1 uses all processors
            self.spectral_clustering.fit(self.features)
            jl.dump(self.spectral_clustering, self.run_dir / f'model.joblib')
        self.labels = self.spectral_clustering.labels_

    def evaluate(self):
        membership_histogram = np.histogram(self.labels, bins=self.run_parameters.num_clusters,
                                            range=(0, self.run_parameters.num_clusters))[0]
        try:
            with open(self.run_dir/'results.json', 'r') as results_file:
                results = json.load(results_file)
        except FileNotFoundError:
            results = {
                'membership_histogram': list(int(n) for n in membership_histogram),
                'median_cluster_coverage': round((np.median(membership_histogram)/len(self.features)), 2),
                'max_cluster_coverage': round((membership_histogram.max() / len(self.features)), 2),
                'min_cluster_coverage': round((membership_histogram.min() / len(self.features)), 2),
                'calinski_harabasz_score': calinski_harabasz_score(self.features, self.labels),
                'davies_bouldin_score': davies_bouldin_score(self.features, self.labels),
                'silouhette_score': silhouette_score(self.features, self.labels),
            }
            with open(self.run_dir / 'results.json', 'w') as results_file:
                json.dump(results, results_file)
        return results

    def select_best(self, results):
        parameters_values, silhouette_scores = [], []
        for parameters, results_values in results.items():
            parameters_values.append(parameters)
            silhouette_scores.append(results_values['silouhette_score'])
        return parameters_values[np.argmax(silhouette_scores)]

    def save_results(self, results):
        formatted_results = {self.format_parameters_key(parameters): value for parameters, value in results.items()}
        with open(self.save_dir / f'{self.name().lower()}_results_{str(datetime.now())[:10]}.json', 'w') as results_file:
            json.dump(formatted_results, results_file)








