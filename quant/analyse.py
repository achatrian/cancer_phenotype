from pathlib import Path
import argparse
import time
import json
import multiprocessing as mp
import pandas as pd
from base.data.wsi_reader import WSIReader
from base.utils import utils
from quant import read_annotations, annotations_summary, find_overlap, ContourProcessor
from quant.features import region_properties, red_haralick, green_haralick, blue_haralick, gray_haralick, \
    surf_points, gray_cooccurrence, orb_descriptor
from quant.graph import HistoGraph


r"""Script with tasks to transform and crunch data"""


def quantify(args):
    r"""Extract features from annotated images in data dir"""
    print(f"Quantifying annotated image data in {str(args.data_dir)} ...")
    contour_struct = read_annotations(args.data_dir)
    annotations_summary(contour_struct)
    opt = WSIReader.get_reader_options(include_path=False)
    label_values = {'epithelium': 200, 'lumen': 250}
    feature_dir = args.data_dir/'data'/'features'
    utils.mkdir(feature_dir)
    average_processing_time = 0.0
    for i, annotation_path in enumerate((Path(args.data_dir)/'data'/'annotations').iterdir()):
        if not annotation_path.is_dir() and annotation_path.suffix == '.json':
            slide_id = annotation_path.name[:-5]
            print(f"Processing {slide_id} ...")
            overlap_struct, contours, contour_bbs, labels = find_overlap(contour_struct[slide_id])
            slide_path = args.data_dir/(slide_id + args.slide_format)
            if not slide_path.is_file():
                raise FileNotFoundError(f"No slide file at {str(slide_path)}")
            reader = WSIReader(opt, slide_path)
            start_processing_time = time.time()
            processor = ContourProcessor((contours, labels), overlap_struct, contour_bbs, label_values, reader,
                                         features=[
                                                 region_properties,
                                                 red_haralick,
                                                 green_haralick,
                                                 blue_haralick,
                                                 gray_haralick,
                                                 surf_points,
                                                 gray_cooccurrence,
                                                 orb_descriptor
                                             ])
            x, data, dist = processor.get_features()
            # below: 'index' yields larger files than 'split', but then the dict with extra custom fields can be loaded
            feature_data = json.loads(x.to_json(orient='index'))  # string must be interpreted
            feature_data['slide_id'] = slide_id
            feature_data['data'] = data
            feature_data['dist'] = dist.tolist()
            with open(feature_dir/(slide_id + '.json'), 'w') as feature_file:
                json.dump(feature_data, feature_file)
            average_processing_time += (time.time() - start_processing_time - average_processing_time) / (i + 1)
            print(f"{slide_id} was processed. Average processing time: {average_processing_time:.2f}")


def merge_features(args):
    r"""Merge all feature files for one dataset into a single feature file for ease of processing"""
    frames, slide_ids = [], []
    feature_dir = Path(args.data_dir) / 'data' / 'features'
    for i, feature_path in enumerate(feature_dir.iterdir()):
        if feature_path.suffix == '.json':
            with open(feature_path, 'r') as feature_file:
                feature_data = json.load(feature_file)
            slide_id = feature_data['slide_id']
            slide_ids.append(slide_id)
            del feature_data['data'], feature_data['dist']
            x = pd.DataFrame.from_dict(feature_data, orient='split')
            frames.append(x)
    combined_x = pd.concat(frames, keys=slide_ids)  # creates multi-index with keys as highest level
    utils.mkdir(feature_dir/'combined')
    with open(feature_dir/'combined'/'features.json', 'w') as features_file:
        combined_x.to_json(features_file, orient='split')  # in 'split' no strings are replicated, thus saving space


def cluster(args):
    r"""Cluster combined features"""
    features_path = Path(args.data_dir) / 'data' / 'features' / 'combined' / 'features.json'
    with open(features_path, 'r') as features_file:
        combined_x = pd.read_json(features_path, orient='skip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True, help="Directory storing the WSIs + annotations")
    parser.add_argument('--slide_format', type=str, default='.ndpi')
    parser.add_argument('--task', type=str, default='quantify', choices=['quantify', 'merge_features'])
    args = parser.parse_args()
    if args.task == 'quantify':
        quantify(args)
    elif args.task == 'merge_feat   ures':
        merge_features(args)
    else:
        raise ValueError(f"Unknown task type {args.task}.")
