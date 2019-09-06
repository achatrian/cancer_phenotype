from pathlib import Path
import argparse
import re
from .tasklog import CheckResult, TaskLog
from data.images.dzi_io import DZI_IO


def completion_test(image_path, tasklog):
    tasklog.match_names()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    opt = parser.parse_args()
    tasklog = TaskLog('process_dzi', opt.data_dir, completion_test)
    tasklog.completion_check()


