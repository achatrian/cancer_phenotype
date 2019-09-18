from pathlib import Path
from typing import Union, Callable
from collections import namedtuple, Counter
from datetime import datetime
import re
import pandas as pd
from tqdm import tqdm

r"""Class for testing extent of task success on given dataset and return compact dataviz representation of progress"""


CheckResult = namedtuple('CheckResult', ['name', 'outcome', 'progress', 'log', 'time'])

MatchResult = namedtuple('MatchResult', ['source_suffix', 'target_suffix', 'matches', 'unmatched', 'num_files', 'num_matches'])


class Log:
    logger = 'dummy class to annotate completion_test'


class TaskLog(Log):
    r"""Use when defining tasks on a dataset in order to test if task was completed on whole dataset"""
    def __init__(self, task_name: str, data_dir: Union[str, Path], completion_test: Callable[[Path, Log], CheckResult]):
        self.task_name = task_name
        self.data_dir = Path(data_dir)
        suffixes = {'.ndpi', '.svs', '.tiff', '.dzi', 'json'}
        self.image_paths = list(path for path in self.data_dir.iterdir() if path.suffix in suffixes)
        if not self.image_paths:
            for suffix in suffixes:
                self.image_paths += list(path for path in self.data_dir.glob(f'*/*{suffix}'))
        if not self.image_paths:
            raise ValueError(f"No suitable image files (.ndpi/.svs'/.tiff) in {str(self.data_dir)}")
        self.completion_test = completion_test
        self.match_result = None  # name matching between two folders

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
            (Path(self.data_dir)/'data'/'logs').mkdir(exist_ok=True, parents=True)
            save_path = Path(self.data_dir)/'data'/'logs'/f'tasklog_{self.task_name}_{str(datetime.now())[:10]}.tsv'
        elif save_path.suffix != '.tsv':
            raise ValueError(f"Save path must point to .tsv file (given extension: {save_path.suffix})")
        results_frame.to_csv(save_path, sep='\t')

    def match_names(self, target_dir, source_suffix=None):
        r"""Match files in two directories by checking whether name of file in data_dir (stripped of suffix) is
        contained in any of the filenames inside the target dir2"""
        paths = list(path for path in Path(self.data_dir).iterdir() if path.is_file())
        # check one level deep
        for dir_path in Path(self.data_dir).iterdir():
            if not dir_path.is_dir():
                continue
            paths += list(path for path in dir_path.iterdir() if path.is_file())
        if source_suffix is None:  # use most common suffix
            suffixes = tuple(path.suffix for path in paths if path.suffix in {'.ndpi', '.svs', '.dzi', '.tiff'})
            source_suffix = Counter(suffixes).most_common(1)[0][0] if suffixes else None
        if source_suffix is not None:
            names_to_match = tuple(path.with_suffix('').name for path in paths if path.suffix == source_suffix)
        else:
            names_to_match = tuple(path.with_suffix('').name for path in paths)
        matches = dict()
        unmatched = []
        for name in names_to_match:
            try:
                match = next(path for path in Path(target_dir).iterdir() if path.is_file() and bool(re.search(name, path.name)))
                matches[name] = match
            except StopIteration:
                unmatched.append(name)
        match_result = MatchResult(
            source_suffix=source_suffix,
            target_suffix=Counter(tuple(path.suffix for path in matches.values())).most_common(1)[0][0],
            matches=matches,
            unmatched=unmatched,
            num_files=len(names_to_match),
            num_matches=len(matches)
        )
        self.match_result = match_result
        return match_result









