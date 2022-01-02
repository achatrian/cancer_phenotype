from pathlib import Path
from datetime import datetime
import json
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.cluster import SpectralClustering, KMeans # spectral clustering implementation in scikit-learn uses K-means for initialisation
from quant.experiment import BaseExperiment  # should inherit from clustering instead ?
from quant.utils.consensus_clustering import ConsensusCluster
from base.utils.utils import bytes2human


class ConsensusExperiment(BaseExperiment):
    r"""This class works with any clustering algorithm class that has
    a n_clusters parameter and a fit_predict method"""

    def __init__(self, args):
        super().__init__(args)
        self.features, self.inliers = (None,) * 2
        self.save_dir = self.args.data_dir / 'data' / 'experiments' / f'{self.name().split("Experiment")[0].lower()}'
        self.save_dir.mkdir(exist_ok=True, parents=True)
        parameters_args_container = {key: (value if type(value) in {str, int, float, bool} else str(value)) for
                                     key, value in
                                     vars(self.args).items() if key in ConsensusExperiment.parameters_names or
                                     key in ConsensusExperiment.preprocess_parameters_names}
        self.results_key = json.dumps(parameters_args_container)
        self.results_file = None
        self.log = {'date': str(datetime.now())}
        self.run_key = None
        self.decomposed, self.reconstruction_error, self.components, self.run_iters = (None,) * 4
        self.consensus_clusterer = None
        self.labels = None

    @staticmethod
    def name():
        return "ConsensusExperiment"

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
        parser.add_argument('--min_clusters_num', type=int, default=5)
        parser.add_argument('--max_clusters_num', type=int, default=10)
        parser.add_argument('--num_resamplings', type=int, default=3, help="Number of resamplings carried out per number of clusters")
        parser.add_argument('--subsample_percentage', type=float, default=0.4, help="Subsample the whole dataset to allow for ")
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
        # sample a subset of the whole dataset
        if len(self.features) > 10000:
            try:  # save a list of sampled datapoints
                with open(self.save_dir / f'subsample_indices.json', 'r') as subsample_indices_file:
                    sample_indices = json.load(subsample_indices_file)
            except FileNotFoundError:
                sample_indices = np.random.randint(0, len(self.features),
                                                   size=round(len(self.features)*self.args.subsample_percentage)).tolist()
                with open(self.save_dir/f'subsample_indices.json', 'w') as subsample_indices_file:
                    json.dump(sample_indices, subsample_indices_file)
            self.features = self.features.iloc[sample_indices]
        self.consensus_clusterer = ConsensusCluster(SpectralClustering, self.args.min_clusters_num,
                                                    self.args.max_clusters_num, self.args.num_resamplings)
        self.consensus_clusterer.fit(self.features)
        self.labels = self.consensus_clusterer.predict()

    def evaluate(self):
        pass

    def select_best(self, results):
        pass

    def save_results(self, results):
        pass








