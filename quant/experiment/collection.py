from pathlib import Path
from typing import Union
from copy import deepcopy
import warnings
from tqdm import tqdm
import pandas as pd
from . import BaseExperiment


class Collection(BaseExperiment):

    def __init__(self, name, steps, step_names=(), experiments=()):
        if experiments:
            step_names = set(experiments[0].named_steps.keys())
            for e in experiments:
                if set(e.named_steps.keys()) != step_names:
                    raise ValueError(f"Collection step names must match ({{{set(e.named_steps.keys())}}} != {step_names})")
        elif not step_names:
            raise ValueError("Either step_names or experiments must be non-empty")
        if len(step_names) != len(steps):
            raise ValueError(F"Number of names is different from number of steps ({len(step_names)} != {len(steps)})")
        self.pipelines = tuple(e.pipeline for e in experiments)
        super().__init__(name, experiments[0].step_names, steps)  # as set() does not preserve order (?)
        self.experiments = experiments
        self.steps = steps

    def read_data_from_dir(self, feature_dir: Union[str, Path], max_memory_use=1e10):
        r"""Overwrite to create one experiment per path !!!"""
        feature_dir = Path(feature_dir)
        feature_paths = sorted((path for path in feature_dir.iterdir() if path.is_file() and path.suffix == '.json'),
                               key=str)
        suffixes = ['th', 'st', 'nd']
        for i, feature_path in enumerate(tqdm(feature_paths)):
            try:
                with open(feature_path, 'r') as features_file:
                    x = pd.read_json(features_file, orient='split')
                self.experiments.append(BaseExperiment(feature_path.name, deepcopy(self.step_names), deepcopy(self.steps)))
                self.experiments[-1].x = x
                self.experiments[-1].loaded_paths.append(feature_path)
                used_memory = sum(e.x.memory_usage(deep=True).sum() for e in self.experiments)
                tqdm.write(f"Memory usage: {used_memory}")
                suffix = suffixes[i % 10] if i % 10 in (1, 2) else suffixes[0]
                if used_memory > max_memory_use:
                    warnings.warn(f"Breaking after {i}{suffix} iteration as memory usage exceeded given maximum ({used_memory} > {max_memory_use})")
                    break
            except ValueError as err:
                print(f"Could not load {str(feature_path)} ...")

