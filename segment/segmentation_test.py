from pathlib import Path
from json import load, dump
from argparse import ArgumentParser
from datetime import datetime
from imageio import imread, imwrite
import numpy as np
import pandas as pd
import cv2
from skimage.transform import rescale
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from tqdm import tqdm
from tifffile import TiffFileError
from openslide.lowlevel import OpenSlideError, OpenSlideUnsupportedFormatError
image_specific_errors = (OpenSlideError, OpenSlideUnsupportedFormatError, TiffFileError)
from data.images.wsi_reader import make_wsi_reader
from base.utils import debug


r"""
Given a set of images and ground truth masks, read the corresponding regions from existing segmentation slides
and compute pixel-wise segmentation metrics for the whole dataset. In addition, create a visualisation image showing
the difference between the network segmentation mask and the ground truth mask.
"""


label_values = {'background': 0, 'epithelium': 200, 'lumen': 250}
label_interval_map = {'epithelium': (70, 225), 'lumen': (225, 255), 'background': (0, 70)}
EPS = 0.0001


def findContours(mask):
    if int(cv2.__version__.split('.')[0]) == 3:
        _, contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # cv2.CHAIN_APPROX_TC89_KCOS)
    return contours, h


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('segmentation_experiment', type=str)
    parser.add_argument('outer_layer', type=str)
    parser.add_argument('--split_path', type=Path, default=None)
    args = parser.parse_args()
    tiles_dir = args.data_dir/'data'/'tiles'/args.experiment_name/args.outer_layer
    if not tiles_dir.exists():
        tiles_dir = args.data_dir/'data'/'tiles'/args.experiment_name
    logs_dir = tiles_dir/'logs'
    slides_masks_dir = args.data_dir/'data'/'masks'/args.segmentation_experiment
    global_results = {}
    global_results['slides'] = slides_results = {}
    slide_tiles_dirs = list(tiles_dir.iterdir())
    if args.split_path:
        with open(args.split_path, 'r') as split_file:
            split = load(split_file)
        try:
            train_slides = set(split['train_slides'])
            test_slides = set(split['test_slides'])
        except KeyError:
            train_slides = set(split['train'][0])
            test_slides = set(split['test'][0])
        train_slides = set.intersection(set(path.name for path in slide_tiles_dirs), train_slides)
        test_slides = set.intersection(set(path.name for path in slide_tiles_dirs), test_slides)
    else:
        train_slides, test_slides = set(), set()
    global_results['train_slides'], global_results['test_slides'] = list(train_slides), list(test_slides)
    for slide_tiles_dir in tqdm(slide_tiles_dirs):
        slide_id = slide_tiles_dir.name
        try:
            with open(logs_dir/(slide_id + '_tiles.json'), 'r') as log_file:
                log = load(log_file)
        except FileNotFoundError:
            print(f"No log for slide {slide_id}")
            continue  # TODO continue or guess mpp?
        try:
            slide_path = slides_masks_dir/f'{slide_id}.tiff'
            slide = make_wsi_reader(slide_path, {'mpp': log['mpp']}, set_mpp=log['mpp'])  # ASSUMING SET_MPP IS THE SAME AS MPP TILES WERE READ AT
        except image_specific_errors:
            print(f"No slide '{slide_id} in {args.data_dir}")
            continue
        images_paths = list(slide_tiles_dir.glob('*.png'))
        if len(images_paths) == 0:
            print(f"No images for slide {slide_id}")
        tiles_average = {'num_tiles': len(images_paths) // 2}
        slides_results[slide_id] = result = {}
        num_objects = 0
        while len(images_paths) >= 2:  # there may be multiple annotations ROIs in a slide
            try:
                image_path = next(path for path in images_paths if path.stem.endswith('__image'))  # two underscores -> only mask
                images_paths.remove(image_path)
            except StopIteration:
                print(f"No tile image for slide {slide_id}")
                continue
            mask_path = image_path.parent/(image_path.stem[:-7] + '__mask.png')
            if not mask_path.exists():
                print(f"No mask image for slide {slide_id}")
                continue
            images_paths.remove(mask_path)
            ground_truth = imread(mask_path)
            experiment_name, x, y, w, h, _, end = image_path.stem.split('_')
            x, y, w, h = int(x), int(y), int(w), int(h)
            rescale_factor = (slide.mpp_x*slide.level_downsamples[slide.read_level])/log['mpp']
            read_w, read_h = round(rescale_factor*w), round(rescale_factor*h)
            mask = slide.read_region((x, y), slide.read_level, (read_w, read_h))
            if rescale_factor != 1.0:
                mask = rescale(mask, 1/rescale_factor, order=0, preserve_range=True)
            for label, interval in label_interval_map.items():
                mask[np.logical_and(interval[0] <= mask, mask <= interval[1])] = label_values[label]
            mask = mask[:ground_truth.shape[0], :ground_truth.shape[1], 0]
            assert set(np.unique(mask)) <= set(label_values.values()), "mask values must match template"
            assert set(np.unique(ground_truth)) <= set(label_values.values()), "ground truth values must match template"
            assert mask.shape == ground_truth.shape, "mask and ground truth must have equal dimensions"
            # compute metrics
            result[image_path.stem] = metrics = {}
            for label, value in label_values.items():
                value_matrix = np.equal(ground_truth, value)
                not_value_matrix = np.not_equal(ground_truth, value)
                tp = np.equal(mask, value)[value_matrix].sum()
                fp = np.equal(mask, value)[not_value_matrix].sum()
                tn = np.not_equal(mask, value)[not_value_matrix].sum()
                fn = np.not_equal(mask, value)[value_matrix].sum()
                assert tp + fp + tn + fn == ground_truth.size, "results must sum to the size of the ground truth image"
                label_accuracy = (tp + tn)/(tp + fp + tn + fn)
                label_precision = tp/(tp + fp + EPS)
                label_recall = tp/(tp + fn + EPS)  # sensitivity
                label_specificity = tn/(tn + fp + EPS)
                label_f1 = 2*(label_precision*label_recall)/(label_precision + label_recall + EPS)
                metrics[label] = {
                    'accuracy': label_accuracy,
                    'precision': label_precision,
                    'recall': label_recall,
                    'specificity': label_specificity,
                    'f1': label_f1
                }
            metrics['balanced_accuracy'] = balanced_accuracy_score(ground_truth.flatten(), mask.flatten())
            metrics['adjusted_rand_score'] = adjusted_rand_score(ground_truth.flatten(), mask.flatten())
            # find number of glands
            contours, _ = findContours((ground_truth > 0).astype(np.uint8))
            num_objects += len(contours)
        if result:
            for label in label_values.keys():
                tiles_average[label] = {
                    'accuracy': np.mean([tile_metrics[label]['accuracy'] for tile_metrics in result.values()]),
                    'precision': np.mean([tile_metrics[label]['precision'] for tile_metrics in result.values()]),
                    'recall': np.mean([tile_metrics[label]['recall'] for tile_metrics in result.values()]),
                    'specificity': np.mean([tile_metrics[label]['specificity'] for tile_metrics in result.values()]),
                    'f1': np.mean([tile_metrics[label]['f1'] for tile_metrics in result.values()])
                }
            tiles_average['balanced_accuracy'] = np.mean([tile_metrics['balanced_accuracy'] for tile_metrics in result.values()])
            tiles_average['adjusted_rand_score'] = np.mean([tile_metrics['adjusted_rand_score'] for tile_metrics in result.values()])
            tiles_average['num_objects'] = num_objects
        else:
            tiles_average = None
        result['slide'] = tiles_average
    date_string = str(datetime.now())[:10]
    results_path = args.data_dir/'data'/'results'/f'segmentation_test_{args.segmentation_experiment}_{args.experiment_name}_{date_string}.json'
    results_path.parent.mkdir(exist_ok=True)
    global_results['dataset'] = {'num_slides': len(global_results['slides'])}
    for label in label_values.keys():
        global_results['dataset'][label] = {
            'accuracy': np.mean([slide_metrics['slide'][label]['accuracy']
                                    for slide_metrics in global_results['slides'].values() if slide_metrics['slide'] is not None]),
            'precision': np.mean([slide_metrics['slide'][label]['precision']
                                     for slide_metrics in global_results['slides'].values() if slide_metrics['slide'] is not None]),
            'recall': np.mean([slide_metrics['slide'][label]['recall']
                                  for slide_metrics in global_results['slides'].values() if slide_metrics['slide'] is not None]),
            'specificity': np.mean([slide_metrics['slide'][label]['specificity']
                                       for slide_metrics in global_results['slides'].values() if slide_metrics['slide'] is not None]),
            'f1': np.mean([slide_metrics['slide'][label]['f1']
                              for slide_metrics in global_results['slides'].values() if slide_metrics['slide'] is not None])
        }
    global_results['dataset']['balanced_accuracy'] = np.mean([slide_metrics['slide']['balanced_accuracy']
                                           for slide_metrics in global_results['slides'].values() if slide_metrics['slide'] is not None])
    global_results['dataset']['adjusted_rand_score'] = np.mean([slide_metrics['slide']['adjusted_rand_score']
                                                              for slide_metrics in global_results['slides'].values() if slide_metrics['slide'] is not None])
    with open(results_path, 'w') as results_file:
        dump(global_results, results_file, allow_nan=False)
    print(f"Saved results for {len(global_results['slides'])}")
    # create excel file with results
    data = []
    for slide_id, slide_metrics in global_results['slides'].items():
        if slide_id in train_slides:
            phase = 'train'
        elif slide_id in test_slides:
            phase = 'test'
        else:
            phase = 'test'  # if no split assume all slides are test
        row = {'slide_id': slide_id, 'phase': phase}
        if slide_metrics['slide'] is None:
            print(f"{slide_id} has no metrics")
            continue
        for label in slide_metrics['slide']:
            if isinstance(slide_metrics['slide'][label], dict):
                for metric in slide_metrics['slide'][label]:
                    row[f'{label[0]}_{metric[0]}'] = round(slide_metrics['slide'][label][metric], 2)
            else:
                if f'{label[0]}' not in row:
                    row[f'{label[0]}'] = round(slide_metrics['slide'][label], 2)
                else:
                    row[f'{label[:5]}'] = round(slide_metrics['slide'][label], 2)
        data.append(row)
    dataframe = pd.DataFrame(data)
    (args.data_dir/'data'/'documents').mkdir(exist_ok=True)
    dataframe.to_excel(args.data_dir/'data'/'documents'/f'segmentation_results_{args.segmentation_experiment}__{args.experiment_name}__{date_string}.xlsx')
    print(f"Saved results spreadsheet in {args.data_dir/'data'/'documents'}")

