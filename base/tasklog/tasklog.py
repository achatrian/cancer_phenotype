from pathlib import Path
from typing import Union, Callable
from collections import namedtuple, Counter
from datetime import datetime
import re
import pandas as pd
from tqdm import tqdm

r"""Class for testing extent of task success on given dataset and returns compact dataviz representation of progress"""


CheckResult = namedtuple('CheckResult', ['name', 'outcome', 'progress', 'log', 'time'])


class Log:
    logger = 'dummy class to annotate completion_test'


class TaskLog(Log):
    r"""Use when defining tasks on a dataset in order to test if task was completed on whole dataset"""
    def __init__(self, task_name: str, data_dir: Union[str, Path], completion_test: Callable[[Path, Log], CheckResult]):
        self.task_name = task_name
        self.data_dir = Path(data_dir)
        suffixes = ('.ndpi', '.svs', '.tiff')
        self.image_paths = list(path for path in self.data_dir.iterdir() if path.suffix in suffixes)
        if not self.image_paths:
            for suffix in suffixes:
                self.image_paths += list(path for path in self.data_dir.glob(f'*/*{suffix}'))
        if not self.image_paths:
            raise ValueError(f"No suitable image files (.ndpi/.svs'/.tiff) in {str(self.data_dir)}")
        self.completion_test = completion_test

    def completion_check(self, save_path=None):
        check_results = []
        print(f"Running completion tests on {len(self.image_paths)} images in {self.data_dir} ...")
        for image_path in tqdm(self.image_paths):
            check_results.append(self.completion_test(image_path, self))  # must use list for pandas
        if not isinstance(check_results[0], CheckResult):
            raise ValueError(f"completion_test must return a CheckResult object (returned {type(check_results[0])})")
        results_frame = pd.DataFrame(data=check_results)
        print(results_frame)
        if save_path is None:
            save_path = Path(self.data_dir)/'logs'/f'progress_log_{self.task_name}_{str(datetime.now())[:10]}.tsv'
        elif save_path.suffix != '.tsv':
            raise ValueError(f"Save path must point to .tsv file (given extension: {save_path.suffix})")
        results_frame.to_csv(save_path, sep='\t')

    def match_names(self, target_dir, suffix=None):
        r"""Match files in two directories by checking whether name of file in data_dir (stripped of suffix) is
        contained in any of the filenames inside the target dir2"""
        if suffix is None:
            suffix = Counter(tuple(path.suffix for path in Path(self.data_dir).iterdir())).most_common(1)[0][0]
        names_to_match = list(path.with_suffix('').name for path in Path(self.data_dir).iterdir() if path.suffix == suffix)
        matches = dict()
        unmatched = []
        for name in names_to_match:
            try:
                match = next(path for path in Path(target_dir).iterdir() if bool(re.search(name, path.name)))
                matches[name] = match
            except StopIteration:
                unmatched.append(name)
        return {
            'suffix': suffix,
            'matches': matches,
            'unmatched': unmatched,
            'num_files': len(names_to_match),
            'num_matched': len(matches)
        }










