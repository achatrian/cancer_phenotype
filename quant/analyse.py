from pathlib import Path
import logging
from datetime import datetime
import argparse
import sys
import time
import json
import multiprocessing as mp
import numpy as np
import pandas as pd
import tqdm
from base.data.wsi_reader import WSIReader
from base.utils import utils
from quant import read_annotations, annotations_summary, find_overlap, ContourProcessor
from quant.features import region_properties, red_haralick, green_haralick, blue_haralick, gray_haralick, \
    surf_points, gray_cooccurrence, orb_descriptor
from quant.graph import HistoGraph


r"""Script with tasks to transform and crunch data"""


def extract_features(annotation_path, feature_dir, contour_struct, label_values, args):
    r"""Quantify features from one annotated image. Used in quantify()"""
    logger = logging.getLogger(__name__)
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
                                     #red_haralick,
                                     #green_haralick,
                                     #blue_haralick,
                                     gray_haralick,
                                     surf_points,
                                     gray_cooccurrence,
                                     #orb_descriptor
                                 ])
    try:
        x, data, dist = processor.get_features()
        # below: 'index' yields larger files than 'split', but then the dict with extra custom fields can be loaded
        with open(feature_dir / (slide_id + '.json'), 'w') as feature_file:
            x.to_json(feature_file, orient='split')
        with open(feature_dir / 'data' / ('data_' + slide_id + '.json'), 'w') as data_file:
            json.dump(data, data_file)
        with open(feature_dir / 'relational' / ('dist_' + slide_id + '.txt'), 'w') as dist_file:
            np.savetxt(dist_file, dist)
            processing_time = time.time() - start_processing_time
            logger.info(f"{slide_id} was processed. Processing time: {processing_time:.2f}s")
    except Exception:
        logger.error(f"Could not process {annotation_path.name} ...")
        logger.error('Failed.', exc_info=True)
    processing_time = time.time() - start_processing_time
    return processing_time


def quantify(args):
    r"""Task: Extract features from annotated images in data dir"""
    print(f"Quantifying annotated image data in {str(args.data_dir)} (workers = {args.workers}) ...")
    contour_struct = read_annotations(args.data_dir)
    annotations_summary(contour_struct)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    (args.data_dir/'logs').mkdir(exist_ok=True)
    fh = logging.FileHandler(args.data_dir/'logs'/f'analyse_quantify_{datetime.now()}.log')
    ch = logging.StreamHandler()  # logging to console for general runtime info
    ch.setLevel(logging.DEBUG)  # if this is set to ERROR, then errors are not printed to log, and vice versa
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter), ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    label_values = {'epithelium': 200, 'lumen': 250}
    feature_dir = args.data_dir/'data'/'features'
    utils.mkdir(feature_dir)  # for features
    utils.mkdir(feature_dir/'data')  # for other data
    utils.mkdir(feature_dir/'relational')  # for relational data between instances
    # skip path if feature file already exists, unless overwrite is passed
    paths = list(path for path in (Path(args.data_dir)/'data'/'annotations').iterdir()
                 if (args.overwrite or not (feature_dir/path.name).is_file()) and
                 not path.is_dir() and path.suffix == '.json')
    logger.info(f"Processing {len(paths)} annotations (overwrite = {args.overwrite}) ...")
    if paths:
        if args.workers > 0:
            with mp.pool.Pool(processes=args.workers) as pool:
                processing_times = pool.starmap(
                    extract_features,
                    [(path, feature_dir, contour_struct, label_values, args) for path in paths]
                )
        else:  # run sequentially
            processing_times = []
            for path in paths:
                processing_times.append(extract_features(path, feature_dir, contour_struct, label_values, args))
        logger.info(f"Quantified {len(processing_times)} in {sum(processing_times)/len(processing_times):.2f}s")
    else:
        logger.info(f"No annotations to process (overwrite = {args.overwrite}).")


def merge_features(args):
    r"""Task: Merge all feature files for one dataset into a single feature file for ease of processing"""
    frames, slide_ids, image_data, distances = [], [], dict(), dict()
    feature_dir = Path(args.data_dir) / 'data' / 'features'
    feature_paths = list(feature_dir.iterdir())
    print("Loading features for each slide ...")
    for i, feature_path in enumerate(tqdm.tqdm(feature_paths)):
        if feature_path.suffix == '.json':
            with open(feature_path, 'r') as feature_file:
                x = pd.read_json(feature_file, orient='split', convert_axes=False, convert_dates=False)
            # with open(feature_path.parent/'data'/('data_' + feature_path.name), 'r') as data_file:
            #     data = json.load(data_file)
            # with open(feature_path.parent/'relational'/('dist_' + feature_path.name[:-4] + '.txt'), 'r') as dist_file:
            #     dist = np.loadtxt(dist_file)
            slide_ids.append(feature_path.name[:-4])
            # image_data[slide_ids[-1]] = data
            # distances[slide_ids[-1]] = dist.tolist()
            frames.append(x)
    utils.mkdir(feature_dir/'combined')
    combined_x = pd.concat(frames, keys=slide_ids)  # creates multi-index with keys as highest level
    with open(feature_dir/'combined'/'features.json', 'w') as features_file:
        combined_x.to_json(features_file, orient='split')  # in 'split' no strings are replicated, thus saving space
    print("Combined feature file was saved ...")
    # with open(feature_dir/'combined'/'image_data.json', 'w') as image_data_file:
    #     json.dump(image_data, image_data_file)
    # print("Combined image data was saved ...")
    # with open(feature_dir/'combined'/'distances.json', 'w') as distances_file:
    #     json.dump(distances, distances_file)
    # print("Combined distance data was saved ...")
    print("Done !")


def cluster(args):
    r"""Cluster combined features"""
    features_path = Path(args.data_dir) / 'data' / 'features' / 'combined' / 'features.json'
    with open(features_path, 'r') as features_file:
        combined_x = pd.read_json(features_path, orient='skip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='task', help="Name of task to perform")
    parser_quantify = subparsers.add_parser('quantify')
    parser_quantify.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser_quantify.add_argument('--slide_format', type=str, default='.ndpi', help="Format of file to extract image data from (with openslide)")
    parser_quantify.add_argument('--workers', type=int, default=4, help="Number of processes used in parallelized tasks")
    parser_quantify.add_argument('--overwrite', action='store_true', help="Whether to overwrite existing feature files")
    parser_merge_features = subparsers.add_parser('merge_features')
    parser_merge_features.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    # general arguments

    args, unparsed = parser.parse_known_args()
    if args.task is None:
        parser.print_help()
        sys.exit("No task name was given.")
    if args.task == 'quantify':
        quantify(args)
    elif args.task == 'merge_features':
        merge_features(args)
    else:
        raise ValueError(f"Unknown task type {args.task}.")
