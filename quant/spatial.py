from argparse import ArgumentParser
from pathlib import Path
from typing import Sequence, Dict
import networkx as nx
import numpy as np


r"""Construct spatial statistics"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser.add_argument('--experiment_name', type=str, required=True, help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    args = parser.parse_args()
    feature_dir = args.data_dir/'data'/'features'/args.experiment_name





