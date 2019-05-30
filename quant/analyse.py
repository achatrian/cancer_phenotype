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


def extract_features(annotation_path, feature_dir, contour_struct, label_values, args):
    r"""Quantify features from one annotated image. Used in quantify()"""
    opt = WSIReader.get_reader_options(include_path=False)
    slide_id = annotation_path.name[:-5]
    print(f"Processing {slide_id} ...")
    overlap_struct, contours, contour_bbs, labels = find_overlap(contour_struct[slide_id])
    slide_path = args.data_dir / (slide_id + args.slide_format)
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
    with open(feature_dir / (slide_id + '.json'), 'w') as feature_file:
        json.dump(feature_data, feature_file)
    processing_time = time.time() - start_processing_time
    print(f"{slide_id} was processed. Processing time: {processing_time:.2f}")
    return processing_time


def quantify(args):
    r"""Task: Extract features from annotated images in data dir"""
    print(f"Quantifying annotated image data in {str(args.data_dir)} ...")
    contour_struct = read_annotations(args.data_dir)
    annotations_summary(contour_struct)
    label_values = {'epithelium': 200, 'lumen': 250}
    feature_dir = args.data_dir/'data'/'features'
    utils.mkdir(feature_dir)
    paths = list((Path(args.data_dir)/'data'/'annotations').iterdir())
    with mp.pool.Pool(processes=args.workers) as pool:
        processing_times = pool.starmap(
            extract_features,
            [(path, feature_dir, contour_struct, label_values, args)
             for path in paths if not path.is_dir() and path.suffix == '.json']
        )
    print(f"Quantified {len(processing_times)} in {sum(processing_times)/len(processing_times):.2f}s")


def merge_features(args):
    r"""Task: Merge all feature files for one dataset into a single feature file for ease of processing"""
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
    parser.add_argument('--slide_format', type=str, default='.ndpi', help="Format of digital image")
    parser.add_argument('--task', type=str, default='quantify', choices=['quantify', 'merge_features'], help="Task that will be run")
    parser.add_argument('--workers', type=int, default=4, help="Number of processes used in parallelized tasks")
    args = parser.parse_args()
    if args.task == 'quantify':
        quantify(args)
    elif args.task == 'merge_feat   ures':
        merge_features(args)
    else:
        raise ValueError(f"Unknown task type {args.task}.")
