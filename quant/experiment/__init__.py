from typing import Union
from pathlib import Path
import warnings
import time
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import pandas as pd


class Experiment:

    def __init__(self, name, step_names, steps, outlier_removal=IsolationForest(), caching_path=None):
        self.name = name
        self.step_names = step_names
        self.x = None
        self.loaded_paths = []
        self.y = None
        self.pipeline = Pipeline(
            memory=caching_path,
            steps=list((name, step) for name, step in zip(step_names, steps))
        )
        self.outlier_removal = outlier_removal
        self.caching_path = caching_path
        self.x_original = None
        self.fitted = False

    def read_data(self, feature_path: Union[str, Path]):
        feature_path = Path(feature_path)
        with open(feature_path, 'r') as features_file:
            self.x = pd.read_json(features_file, orient='split')
            self.loaded_paths.append(feature_path)

    def read_data_from_dir(self, feature_dir: Union[str, Path], max_memory_use=1e10):
        feature_dir = Path(feature_dir)
        frames = []
        feature_paths = sorted((path for path in feature_dir.iterdir() if path.is_file() and path.suffix == '.json'),
                               key=str)
        suffixes = ['th', 'st', 'nd']
        for i, feature_path in enumerate(tqdm(feature_paths)):
            try:
                with open(feature_path, 'r') as features_file:
                    frames.append(pd.read_json(features_file, orient='split', convert_axes=False, convert_dates=False))
                self.loaded_paths.append(feature_path)
                used_memory = sum(frame.memory_usage(deep=True).sum() for frame in frames)
                tqdm.write(f"Memory usage: {used_memory}")
                suffix = suffixes[i % 10] if i % 10 in (1, 2) else suffixes[0]
                if used_memory > max_memory_use:
                    warnings.warn(f"Breaking after {i}{suffix} iteration as memory usage exceeded given maximum ({used_memory} > {max_memory_use})")
                    break
            except ValueError as err:
                print(f"Could not load {str(feature_path)} ...")
        if len(frames) == 1:
            self.x = frames[0]
        else:
            self.x = pd.concat(frames, keys=tuple((path.with_suffix('')).name for path in self.loaded_paths))
        print(f"{len(self.loaded_paths)} feature files were loaded.")

    def add_steps(self, items, names, overwrite=False):
        if not overwrite:
            self.pipeline.steps.extend((name, item) for item, name in zip(items, names))
        else:
            self.pipeline.steps = list((name, item) for item, name in zip(items, names))
        self.pipeline._validate_steps()

    def remove_outliers(self):
        n_before = len(self.x)
        # below: need to cast to bool or a np boolean is returned + need to use list as tuple is considered a key by []
        inliers = list(bool(label != -1) for label in self.outlier_removal.fit_predict(self.x))
        self.x = self.x.loc[inliers]
        n_after = len(self.x)
        print(f"Removed {n_before - n_after} outliers through {str(self.outlier_removal)}")

    def run(self, keep_preprocessed=True, store_original=False):
        r"""Method where y is computed. Can be overwritten"""
        if self.x is None:
            raise ValueError(f"Data  has not been loaded for '{self.name}'")
        start_time = time.time()
        y = self.pipeline.fit_predict(self.x)
        self.y = pd.DataFrame(data=y, columns=['cluster'], index=self.x.index)
        print(f"Total run time: {time.time() - start_time}s")
        if keep_preprocessed:
            if store_original:
                self.x_original = self.x.copy(deep=True)
            x = self.x.values
            for step_name in self.step_names[:-1]:  # pro
                x = self.pipeline.named_steps[step_name].transform(x)  # transforms have already been fitted
            columns = list(self.step_names[-2] + str(i) for i in range(x.shape[1]))  # e.g. PCA0 PCA1 ..
            self.x = pd.DataFrame(data=x, columns=columns, index=self.x.index)
        self.fitted = True

    def get_subsets(self):
        r"""Assuming x uses a multindex, returns a dataframe for each field in the first level"""
        return {subset_name: self.x[subset_name] for subset_name in self.x.index.levels[0]}

    def save_results(self, save_dir):
        r"""Save experiment results"""
        if '~' in str(save_dir):
            save_dir = Path(save_dir).expanduser()
        xpath = Path(save_dir)/f'x_{self.name}_{"fitted" if self.fitted else ""}.json'
        self.x.to_json(open(xpath, 'w'), orient='split')
        if self.y is not None:
            ypath = Path(save_dir)/f'y_{self.name}.json'
            self.y.to_json(open(ypath, 'w'), orient='split')





