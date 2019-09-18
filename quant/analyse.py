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
from data.images.wsi_reader import WSIReader
from base.utils import utils
from data import read_annotations, annotations_summary, find_overlap
from data.contour_processor import ContourProcessor
from quant.features import region_properties, gray_haralick, \
    surf_points, gray_cooccurrence

r"""Script with tasks to transform and crunch data"""


def extract_features(annotation_path, slide_path, feature_dir, label_values, args):
    r"""Quantify features from one annotated images. Used in quantify()"""
    logger = logging.getLogger(__name__)
    opt = WSIReader.get_reader_options(include_path=False)
    slide_id = annotation_path.with_suffix('').name
    contour_struct = read_annotations(args.data_dir, slide_ids=(slide_id,), experiment_name=args.experiment_name)
    annotations_summary(contour_struct)
    print(f"Processing {slide_id} ...")
    overlap_struct, contours, contour_bbs, labels = find_overlap(contour_struct[slide_id])
    reader = WSIReader(slide_path, opt)
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
        (feature_dir/'data'/args.experiment_name).mkdir(exist_ok=True, parent=True)
        with open(feature_dir/'data'/args.experiment_name/('data_' + slide_id + '.json'), 'w') as data_file:
            json.dump(data, data_file)
        (feature_dir/'relational'/args.experiment_name).mkdir(exist_ok=True, parent=True)
        with open(feature_dir/'relational'/args.experiment_name/('dist_' + slide_id + '.txt'), 'w') as dist_file:
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
    print(f"Quantifying annotated images data in {str(args.data_dir)} (workers = {args.workers}) ...")
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
    feature_dir.mkdir(exist_ok=True)  # for features
    (feature_dir/'data').mkdir(exist_ok=True)  # for other data
    (feature_dir/'relational').mkdir(exist_ok=True)  # for relational data between instances
    # skip path if feature file already exists, unless overwrite is passed
    annotation_paths = sorted((path for path in (Path(args.data_dir)/'data'/'annotations'/args.experiment_name).iterdir()
                 if (args.overwrite or not (feature_dir/path.name).is_file()) and
                 not path.is_dir() and path.suffix == '.json'), key=lambda path: path.with_suffix('').name)
    slide_paths_ = list(path for path in Path(args.data_dir).iterdir()
                       if path.suffix == '.svs' or path.suffix == '.ndpi')
    slide_paths_ += list(path for path in Path(args.data_dir).glob('*/*.ndpi'))
    slide_paths_ += list(path for path in Path(args.data_dir).glob('*/*.svs'))
    slide_paths_ = sorted(slide_paths_, key=lambda path: path.with_suffix('').name)
    slide_paths = []
    # match a slide path to every annotation path. Account for missing annotations by ignoring the corresponding slides
    # assume slides and annotations are in the same order
    i, j = 0, 0
    while len(slide_paths) != len(annotation_paths):
        annotation_path, slide_path = annotation_paths[i], slide_paths_[j]
        if slide_path.with_suffix('').name.startswith(annotation_path.with_suffix('').name):
            slide_paths.append(slide_path)
            i += 1
        j += 1
    logger.info(f"Processing {len(annotation_paths)} annotations (overwrite = {args.overwrite}) ...")
    if annotation_paths:
        if args.workers > 0:
            with mp.pool.Pool(processes=args.workers) as pool:
                processing_times = pool.starmap(
                    extract_features,
                    [(annotation_path, slide_path, feature_dir, label_values, args)
                     for annotation_path, slide_path in zip(annotation_paths, slide_paths)]
                )
        else:  # run sequentially
            processing_times = []
            for annotation_path, slide_path in zip(annotation_paths, slide_paths):
                processing_times.append(extract_features(annotation_path, slide_path, feature_dir, label_values, args))
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
    (feature_dir/'combined').mkdir(exist_ok=True)
    combined_x = pd.concat(frames, keys=slide_ids)  # creates multi-index with keys as highest level
    with open(feature_dir/'combined'/'features.json', 'w') as features_file:
        combined_x.to_json(features_file, orient='split')  # in 'split' no strings are replicated, thus saving space
    print("Combined feature file was saved ...")
    # with open(feature_dir/'combined'/'image_data.json', 'w') as image_data_file:
    #     json.dump(image_data, image_data_file)
    # print("Combined images data was saved ...")
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
    parser_quantify.add_argument('--experiment_name', type=str, help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    parser_quantify.add_argument('--slide_format', type=str, default='.ndpi', help="Format of file to extract images data from (with openslide)")
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
