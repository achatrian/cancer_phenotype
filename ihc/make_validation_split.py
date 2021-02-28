from pathlib import Path
from datetime import datetime
import json
from base.options.base_options import BaseOptions
from ihc.datasets.ihcpatch_dataset import IHCPatchDataset


if __name__ == '__main__':
    opt = BaseOptions().gather_options()
    opt.is_apply = True
    assert opt.dataset_name == 'ihcpatch'
    dataset = IHCPatchDataset(opt)
    validation_dir = Path('/well/rittscher/projects/IHC_Request/validation_cases')
    validation_slides = tuple(path.with_suffix('').name for path in validation_dir.iterdir())
    cv_path = Path(opt.data_dir)/'data'/'cross_validate'
    cv_path.mkdir(exist_ok=True, parents=True)
    validation_split = {
        'split_num': 0,
        'train_slides': [],
        'train_slide_labels': [],
        'test_slides': validation_slides,
        'test_slide_labels': [],
        'train_paths': [],
        'test_paths': [],
        'train_labels': [],
        'test_labels': [],
        'date': str(datetime.now())[:10],
        'train_fraction': 0.0
    }
    with open(cv_path/f'validation-split.json', 'w') as split_file:
        json.dump(validation_split, split_file)
    print("Done!")