from pathlib import Path
import json
from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold
from base.options.base_options import BaseOptions
from benign_malignant.datasets.benign_malignant_dataset import BenignMalignantDataset


"Split dataset by keeping ratio of slides with more benign glands over those with more malignant glands constant"


if __name__ == '__main__':
    n_splits = 3
    opt = BaseOptions().gather_options()
    assert opt.dataset_name == 'gland'
    cv_path = Path(opt.data_dir)/'data'/'cross_validate'
    cv_path.mkdir(exist_ok=True, parents=True)
    for i in range(n_splits):  # remove existing splits or dataset could break because of paths mismatch
        (cv_path/f'{n_splits}-split{i}.json').unlink()
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
    dataset = BenignMalignantDataset(opt)
    assert dataset.paths
    slide_ids = set(tile_path.parent.name for tile_path in dataset.paths)
    mean_labels, slide_labels = {}, {}
    for slide_id in slide_ids:
        mean_label = sum(dataset.labels[i] for i, path in enumerate(dataset.paths) if path.parent.name == slide_id) / \
                     sum(1 for path in dataset.paths if path.parent.name == slide_id)
        mean_labels[slide_id] = mean_label
        slide_labels[slide_id] = round(mean_label >= 0.5)
    slides = np.array(list(slide_ids))
    labels = np.array([slide_labels[slide] for slide in slides])
    for i, (train_index, test_index) in enumerate(skf.split(slides, labels)):
        train_slides, test_slides = slides[train_index].tolist(), slides[test_index].tolist()
        train_labels, test_labels = labels[train_index].tolist(), labels[test_index].tolist()
        split = {
            'split_num': i,
            'train_slides': train_slides,
            'train_slide_labels': train_labels,
            'train_mean_labels': [mean_labels[slide] for slide in train_slides],
            'test_slides': test_slides,
            'test_slide_labels': test_labels,
            'test_mean_labels': [mean_labels[slide] for slide in test_slides],
            'train_paths': [str(path) for path in dataset.paths if path.parent.name in train_slides],
            'test_paths': [str(path) for path in dataset.paths if path.parent.name in test_slides],
            'train_labels': [label for path, label in zip(dataset.paths, dataset.labels) if path.parent.name in train_slides],
            'test_labels': [label for path, label in zip(dataset.paths, dataset.labels) if path.parent.name in test_slides],
            'date': str(datetime.now())[:10],
            'train_fraction': round((n_splits - 1)/n_splits, 2),
        }
        split['train_mean_label'] = sum(split['train_mean_labels'])/len(split['train_mean_labels'])
        split['test_mean_label'] = sum(split['test_mean_labels']) / len(split['test_mean_labels'])
        with open(cv_path/f'{n_splits}-split{i}.json', 'w') as split_file:
            json.dump(split, split_file)
        print(f"Split {i} has train mean label = {split['train_mean_label']} and test mean label = {split['test_mean_label']}")
    print("Done!")





