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
from tqdm import tqdm
from sklearn.utils import shuffle
import h5py
import joblib as jl
from data.images.wsi_reader import WSIReader
from data.contours import read_annotations, annotations_summary
from data.contours.instance_masker import InstanceMasker
from quant.contour_processor import ContourProcessor
from quant.features import region_properties, gray_haralick, gray_cooccurrence
from base.utils.utils import bytes2human

r"""Script with tasks to transform and crunch data"""


def extract_features(annotation_path, slide_path, feature_dir, label_values, args):
    r"""Quantify features from one annotated images. Used in quantify()"""
    logger = logging.getLogger(__name__)
    opt = WSIReader.get_reader_options(include_path=False)
    slide_id = annotation_path.with_suffix('').name
    logger.info(f"Reading contours for {slide_id} ...")
    contour_struct = read_annotations(args.data_dir, slide_ids=(slide_id,), experiment_name=args.experiment_name)
    annotations_summary(contour_struct)
    logger.info(f"Processing {slide_id} ...")
    reader = WSIReader(slide_path, opt)
    start_processing_time = time.time()
    masker = InstanceMasker(contour_struct[slide_id], args.outer_label, label_values)
    processor = ContourProcessor(masker, reader,
                                 features=[
                                     region_properties,
                                     #red_haralick,
                                     #green_haralick,
                                     #blue_haralick,
                                     gray_haralick,
                                     #surf_points,
                                     gray_cooccurrence,
                                     #orb_descriptor
                                 ])
    try:
        x, data, dist = processor.get_features()
        # below: 'index' yields larger files than 'split', but then the dict with extra custom fields can be loaded
        with open(feature_dir / (slide_id + '.json'), 'w') as feature_file:
            x.to_json(feature_file, orient='split')
        (feature_dir/'data'/args.experiment_name).mkdir(exist_ok=True, parents=True)
        with open(feature_dir/'data'/args.experiment_name/('data_' + slide_id + '.json'), 'w') as data_file:
            json.dump(data, data_file)
        (feature_dir/'relational'/args.experiment_name).mkdir(exist_ok=True, parents=True)
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
    if args.shuffle_annotations:
        annotation_paths, slide_paths = shuffle(annotation_paths, slide_paths)
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


def compress(args):
    r"""Task: Merge all feature files for one dataset into a single feature file for ease of processing"""
    feature_dir = Path(args.data_dir)/'data'/'features'
    compress_path = feature_dir/'compressed.h5'
    feature_paths = list(feature_dir.iterdir())
    i, pre_compression_size = 0, 0
    print("Loading features for each slide ...")
    for i, feature_path in enumerate(tqdm(feature_paths)):
        if feature_path.suffix == '.json':
            with open(feature_path, 'r') as feature_file:
                slide_features = pd.read_json(feature_file, orient='split', convert_axes=False, convert_dates=False)
            pre_compression_size += slide_features.memory_usage().sum()
            slide_id = feature_path.name[:-4]
            slide_features.to_hdf(compress_path, key=slide_id)
    print(f"Done! Features for {i} slides were compressed from {bytes2human(pre_compression_size)} to {compress_path.stat().st_size}!")


def dim_reduce(args):  # TODO test
    from sklearn.decomposition import IncrementalPCA

    r"""Task: Reduce the size of compressed features"""
    feature_dir = Path(args.data_dir) / 'data' / 'features'
    compressed_path, dim_reduce_path = feature_dir/'compressed.h5', feature_dir/'dim_reduced.h5'
    slide_ids = list(h5py.File(compressed_path, 'r').keys())  # read keys to access different stored frames
    slide_featuress = []
    pre_dim_reduce_size, post_dim_reduce_size = 0, 0
    ipca = IncrementalPCA(n_components=args.n_components)
    try:
        if args.overwrite_model:
            raise FileNotFoundError("")
        ipca = jl.load(ipca, feature_dir / 'dim_reduce_ipca_model.joblib')
    except FileNotFoundError:
        # fit loop
        for i, slide_id in enumerate(tqdm(slide_ids, desc=f"Computing PCA incrementally for {len(slide_ids)} slides ...")):
            slide_features = pd.read_hdf(compressed_path, slide_id)
            pre_dim_reduce_size += slide_features.memory_usage().sum()
            ipca = ipca.partial_fit(slide_features)
        jl.dump(ipca, feature_dir/'dim_reduce_ipca_model.joblib')
    # save loop
    for i, slide_id in enumerate(tqdm(slide_ids, desc=f"Dimensionality reduction ...")):
        slide_features = pd.read_hdf(dim_reduce_path, slide_id)
        slide_features = pd.DataFrame(data=ipca.transform(slide_features), index=slide_features.index)
        slide_features.to_hdf(dim_reduce_path, key=slide_id)
        post_dim_reduce_size += slide_features.memory_usage().sum()
    print(f"Done! Dimensionality reduction decreased size from {bytes2human(pre_dim_reduce_size)} to {bytes2human(post_dim_reduce_size)}")

# ########## subparser commands ############


def add_quantify_subparser(subparsers):
    parser_quantify = subparsers.add_parser('quantify')
    parser_quantify.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser_quantify.add_argument('--experiment_name', type=str,
                                 help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    parser_quantify.add_argument('--slide_format', type=str, default='.ndpi',
                                 help="Format of file to extract images data from   (with openslide)")
    parser_quantify.add_argument('--outer_label', type=str, default='epithelium',
                                 help="Label whose instances are of interest for feature quantification")
    parser_quantify.add_argument('--workers', type=int, default=4,
                                 help="Number of processes used in parallelized tasks")
    parser_quantify.add_argument('--shuffle_annotations', action='store_true',
                                 help="Shuffle processing order. Useful if multiple processing jobs are launched at the same time.")
    parser_quantify.add_argument('--overwrite', action='store_true', help="Whether to overwrite existing feature files")
    # parser_quantify.add_argument('--max_contour_num', type=int, default=3000,
    # help="Max number of glands in one slide to sample")


def add_compress_subparser(subparsers):
    parser_compress = subparsers.add_parser('compress')
    parser_compress.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")


def add_dim_reduce_subparser(subparsers):
    parser_dim_reduce = subparsers.add_parser('dim_reduce')
    parser_dim_reduce.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser_dim_reduce.add_argument('--n_components', type=int, default=20, help="Number of principal components in incremental PCA")
    parser_dim_reduce.add_argument('--overwrite_model', action='store_true')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='task', help="Name of task to perform")
    add_quantify_subparser(subparsers)
    add_compress_subparser(subparsers)
    # general arguments
    args, unparsed = parser.parse_known_args()
    if args.task is None:
        parser.print_help()
        sys.exit("No task name was given.")
    if args.task == 'quantify':
        quantify(args)
    elif args.task == 'compress':
        compress(args)
    else:
        raise ValueError(f"Unknown task type {args.task}.")
