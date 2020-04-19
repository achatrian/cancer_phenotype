from pathlib import Path
import json
from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold
from base.options.base_options import BaseOptions
from ihc.datasets.ihcpatch_dataset import IHCPatchDataset


if __name__ == '__main__':
    n_splits = 5
    opt = BaseOptions().gather_options()
    assert opt.dataset_name == 'ihcpatch'
    dataset = IHCPatchDataset(opt)
    slides = np.array(list(slide_id for slide_id, stain in dataset.slide_stains.items() if stain == 'nan'))
    labels = np.array(list(dataset.slide_labels[slide_id] for slide_id in slides))
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
    cv_path = Path(opt.data_dir)/'data'/'cross_validate'
    cv_path.mkdir(exist_ok=True, parents=True)
    for i, (train_index, test_index) in enumerate(skf.split(slides, labels)):
        train_slides, test_slides = slides[train_index].tolist(), slides[test_index].tolist()
        train_labels, test_labels = labels[train_index].tolist(), labels[test_index].tolist()
        split = {
            'split_num': i,
            'train_slides': train_slides,
            'train_slide_labels': train_labels,
            'test_slides': test_slides,
            'test_slide_labels': test_labels,
            'train_paths': [str(path) for path in dataset.paths if path.parent.name in train_slides],
            'test_paths': [str(path) for path in dataset.paths if path.parent.name in test_slides],
            'train_labels': [label for path, label in zip(dataset.paths, dataset.labels) if path.parent.name in train_slides],
            'test_labels': [label for path, label in zip(dataset.paths, dataset.labels) if path.parent.name in test_slides],
            'date': str(datetime.now())[:10],
            'train_fraction': round((n_splits - 1)/n_splits, 2)
        }
        with open(cv_path/f'{n_splits}-split{i}.json', 'w') as split_file:
            json.dump(split, split_file)
    print("Done!")





