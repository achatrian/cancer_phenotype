from pathlib import Path
import argparse
import re
from tasklog import CheckResult, TaskLog
from dzi_io import DZI_IO


def completion_test(image_path, tasklog):
    slide_id = image_path.with_suffix('').name
    dzi_dir = Path(opt.data_dir)/'data'/'dzi'
    source_name = re.sub('\.(ndpi|svs)', '.dzi', slide_id)
    source_name = source_name if source_name.endswith('.dzi') else source_name + '.dzi'
    target_path = Path('masks')/('mask_' + Path(source_name).name)
    try:
        source_dzi = DZI_IO(str(dzi_dir/source_name))
    except FileNotFoundError:
        return CheckResult(str(source_name), False, None, 'Source file is incomplete', None)
    try:
        target_dzi = DZI_IO(str(dzi_dir/target_path))
    except FileNotFoundError:
        return CheckResult(str(target_path.name), False, None, 'Target file is incomplete', None)
    progress = target_dzi.level_count / (source_dzi.level_count - 1)  # high res level is discarded
    outcome = progress == 1.0
    return CheckResult(str(source_name), outcome, progress, '', None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    opt = parser.parse_args()
    tasklog = TaskLog('process_dzi', opt.data_dir, completion_test)
    tasklog.completion_check()


