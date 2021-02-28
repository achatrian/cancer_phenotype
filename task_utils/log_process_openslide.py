from pathlib import Path
import argparse
import re
from .tasklog import CheckResult, TaskLog
from data.images.wsi_reader import WSIReader


def completion_test(image_path, tasklog):
    slide_id = image_path.with_suffix('').name
    try:
        source_dzi = WSIReader(MISSING)
    except FileNotFoundError:
        return CheckResult(str(source_name), False, None, 'Source file is incomplete', None)
    try:
        target_dzi = WSIReader(MISSING)
    except FileNotFoundError:
        return CheckResult(str(source_name), False, None, 'Target file is incomplete', None)
    progress = target_dzi.level_count / (source_dzi.level_count - 1)  # high res level is discarded
    outcome = progress == 1.0
    return CheckResult(str(source_name), outcome, progress, '', None)


if __name__ == '__main__':  # TODO finish
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    opt = parser.parse_args()
    tasklog = TaskLog('process_dzi', opt.data_dir, completion_test)
    tasklog.completion_check()

