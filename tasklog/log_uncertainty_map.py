from pathlib import Path
import argparse
from tasklog import CheckResult, TaskLog


def completion_test(image_path, tasklog):
    slide_id = image_path.with_suffix('').name
    matched_names = tuple(tasklog.match_result.matches.keys())
    unmatched_names = tuple(path.with_suffix('').name for path in tasklog.match_result.unmatched)
    if slide_id in unmatched_names:
        return CheckResult(slide_id, False, None, 'no uncertainty_map', None)
    elif slide_id in matched_names:
        return CheckResult(slide_id, True, None, f'uncertainty map found: {tasklog.match_result.matches[slide_id]}', None)
    else:
        raise KeyError(f"Slide {slide_id} was not found during name match.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    opt = parser.parse_args()
    tasklog = TaskLog('uncertainty_map', opt.data_dir, completion_test)
    tasklog.match_names(Path(opt.data_dir)/'data'/'uncertainty_map')
    tasklog.completion_check()
