from pathlib import Path
from datetime import datetime
import json
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import NMF
from sklearn.ensemble import IsolationForest
from base.utils.utils import bytes2human
from quant.experiment import BaseExperiment  # should inherit from clustering instead ?


# TODO check rational behind using nmf for clustering ? if there is one, TEST
class NMFExperiment(BaseExperiment):
    parameters_names = ('num_components',)
    parameters_defaults = ('30',)
    parameters_descriptions = ('Components extracted by nonnegative matrix factorisation',)

    def __init__(self, args):
        super().__init__(args)
        self.features, self.inliers = (None,) * 2
        self.save_dir = self.args.data_dir/'data'/'experiments'/'self.nmf'
        self.save_dir.mkdir(exist_ok=True, parents=True)
        parameters_args_container = {key: (value if type(value) in {str, int, float, bool} else str(value)) for key, value in
                          vars(self.args).items() if key in NMFExperiment.parameters_names or
                          key in NMFExperiment.preprocess_parameters_names}
        self.results_key = json.dumps(parameters_args_container)
        self.results_file = None
        self.log = {'date': str(datetime.now())}
        self.run_key = None
        self.nmf = None
        self.decomposed, self.reconstruction_error, self.components, self.run_iters = (None,)*4

    @staticmethod
    def name():
        return "NMFExperiment"

    @classmethod
    def modify_commandline_options(cls, parser):
        parser = super().modify_commandline_options(parser)
        parser.add_argument('--segmentation_experiment', type=str)
        parser.add_argument('--features_filename', type=str, default='filtered.h5',
                            help="Basename of files storing features")
        parser.add_argument('--outlier_removal', action='store_true')
        parser.add_argument('--isolation_contamination', default='auto',
                            help="Contamination parameter in the isolation  (auto or float)")
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
        if parameters is None:
            raise ValueError("parameters cannot be None")
        self.run_parameters = parameters
        self.run_key = self.format_parameters_key(self.run_parameters)
        results_group = self.results_file[self.results_key]
        iteration_group = results_group.create_group(self.run_key)
        self.nmf = NMF(parameters.num_components, verbose=True)
        tqdm.write("Factorizing feature matrix ...")
        self.decomposed = pd.DataFrame(data=self.nmf.fit_transform(self.features), index=self.features.index)
        self.reconstruction_error, self.components, self.run_iters = self.nmf.reconstruction_err_, self.nmf.components_, self.nmf.n_iter
        iteration_group.create_dataset('decomposed_features', data=self.decomposed)
        iteration_group.create_dataset('reconstruction_error', data=self.reconstruction_error)
        iteration_group.create_dataset('components', data=self.components)
        iteration_group.create_dataset('num_optimizer_iterations', data=self.run_iters)

    def evaluate(self):
        # TODO add clustering measures
        return {
            'reconstruction_error': self.reconstruction_error,
            'components_norm': np.linalg.norm(self.components),
            'num_optimizer_iterations': self.run_iters
        }


def save_log(log, data_dir):
    (data_dir / 'data' / 'logs').mkdir(exist_ok=True, parents=True)
    with (data_dir / 'data' / 'logs' / f'tcga_analysis_{log["date"]}.json').open('w') as log_file:
        json.dump(log, log_file)


