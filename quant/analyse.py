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
import cv2
from tqdm import tqdm
from imageio import imread
from sklearn.utils import shuffle
import h5py
import joblib as jl
from staintools import StainNormalizer
from data.images.wsi_reader import WSIReader
from data.contours import read_annotations, annotations_summary, check_point_overlap
from data.contours.instance_masker import InstanceMasker
from quant.utils.contour_processor import ContourProcessor
from features import region_properties, gray_haralick, gray_cooccurrence, nuclear_features
from base.utils.utils import bytes2human


r"""Script with tasks to transform and crunch data"""


def extract_features(annotation_path, slide_path, feature_dir, label_values, args):
    r"""Quantify features from one annotated images. Used in quantify()"""
    logger = logging.getLogger(__name__)
    opt = WSIReader.get_reader_options(include_path=False)
    slide_id = annotation_path.with_suffix('').name
    if (feature_dir/(slide_id + '.json')).exists():
        return 0.0
    logger.info(f"Reading contours for {slide_id} ...")
    contour_struct = read_annotations(args.data_dir, slide_ids=(slide_id,), experiment_name=args.experiment_name)
    annotations_summary(contour_struct)
    logger.info(f"Processing {slide_id} ...")
    reader = WSIReader(slide_path, opt)
    start_processing_time = time.time()
    macenko_normalizer = StainNormalizer(method='macenko')
    reference_image = imread(Path(args.data_dir, 'data', f'{args.outer_label}:stain_references',
                                  'references', f'{args.reference_slide}.png'))
    macenko_normalizer.fit(reference_image)
    slide_stain_matrix = np.load(Path(args.data_dir, 'data', f'{args.outer_label}:stain_references',
                                      f'{args.reference_slide}.npy'))
    masker = InstanceMasker(contour_struct[slide_id], args.outer_label, label_values)
    processor = ContourProcessor(masker, reader,
                                 features=[
                                     region_properties,
                                     nuclear_features,
                                     #red_haralick,
                                     #green_haralick,
                                     #blue_haralick,
                                     gray_haralick,
                                     #surf_points,
                                     gray_cooccurrence,
                                     #orb_descriptor
                                 ],
                                 stain_normalizer=macenko_normalizer,
                                 stain_matrix=slide_stain_matrix)
    try:
        x, data, dist = processor.get_features()
        # below: 'index' yields larger files than 'split', but then the dict with extra custom fields can be loaded
        with open(feature_dir / (slide_id + '.json'), 'w') as feature_file:
            x.to_json(feature_file, orient='split')
        (feature_dir/'data').mkdir(exist_ok=True, parents=True)
        with open(feature_dir/'data'/('data_' + slide_id + '.json'), 'w') as data_file:
            json.dump(data, data_file)
        (feature_dir/'relational').mkdir(exist_ok=True, parents=True)
        with open(feature_dir/'relational'/('dist_' + slide_id + '.json'), 'w') as dist_file:
            dist.to_json(dist_file)
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
    label_values = {'epithelium': 200, 'lumen': 250, 'nuclei': 50}
    feature_dir = args.data_dir/'data'/'features'/args.experiment_name
    feature_dir.mkdir(exist_ok=True, parents=True)  # for features
    (feature_dir/'data').mkdir(exist_ok=True)  # for other data
    (feature_dir/'relational').mkdir(exist_ok=True)  # for relational data between instances
    # skip path if feature file already exists, unless overwrite is passed
    annotation_paths = sorted((path for path in (Path(args.data_dir)/'data'/'annotations'/args.experiment_name).iterdir()
                 if (args.overwrite or not (feature_dir/path.name).is_file()) and
                 not path.is_dir() and path.suffix == '.json'), key=lambda path: path.with_suffix('').name)
    if args.debug_slides is not None:  # select subset for debugging
        annotation_paths = [annotation_path for annotation_path in annotation_paths
                            if annotation_path.with_suffix('').name in args.debug_slides]
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
    if args.reference_slide is None:
        args.reference_slide = slide_paths[0].with_suffix('').name
    logger.info(f"Using '{args.reference_slide}' as reference slide for stain normalization")
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
    feature_dir = args.data_dir/'data'/'features'/args.experiment_name
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
    print(f"Done! Features for {i} slides were compressed from {bytes2human(pre_compression_size)} to {bytes2human(compress_path.stat().st_size)}!")


def dim_reduce(args):  # TODO test
    r"""Task: Reduce the size of compressed features, useful to reduce very large datasets in size"""

    from sklearn.decomposition import IncrementalPCA
    from sklearn.preprocessing import StandardScaler
    # FIXME should scale all features before computing PCA, but these should be all at once to get right sscaler parameters
    feature_dir = args.data_dir/'data'/'features'/args.experiment_name
    compressed_path, dim_reduce_path = feature_dir/'compressed.h5', feature_dir/'dim_reduced.h5'
    slide_ids = list(h5py.File(compressed_path, 'r').keys())  # read keys to access different stored frames
    pre_dim_reduce_size, post_dim_reduce_size = 0, 0
    ipca = IncrementalPCA(n_components=args.n_components)
    try:
        if args.overwrite_model:
            raise FileNotFoundError("")
        ipca = jl.load(feature_dir / 'dim_reduce_ipca_model.joblib')
    except FileNotFoundError:
        # fit loop
        for i, slide_id in enumerate(tqdm(slide_ids, desc=f"Computing PCA incrementally for {len(slide_ids)} slides ...")):
            slide_features = pd.read_hdf(compressed_path, slide_id)
            if len(slide_features) < args.n_components:
                continue
            pre_dim_reduce_size += slide_features.memory_usage().sum()
            ipca = ipca.partial_fit(slide_features)
        jl.dump(ipca, feature_dir/'dim_reduce_ipca_model.joblib')
    print(f"{ipca.explained_variance_ratio_}")
    # save loop
    for i, slide_id in enumerate(tqdm(slide_ids, desc=f"Dimensionality reduction ...")):
        slide_features = pd.read_hdf(compressed_path, slide_id)
        slide_features = pd.DataFrame(data=ipca.transform(slide_features), index=slide_features.index)
        slide_features.to_hdf(dim_reduce_path, key=slide_id)
        post_dim_reduce_size += slide_features.memory_usage().sum()
    print(f"Done! Dimensionality reduction decreased size from {bytes2human(pre_dim_reduce_size)} to {bytes2human(post_dim_reduce_size)}")


def filter_(args):
    r"""Task: filter data points"""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import StandardScaler
    feature_dir = args.data_dir/'data'/'features'/args.experiment_name
    compressed_path = feature_dir/'compressed.h5'
    filtered_path = feature_dir/'filtered.h5'
    slide_ids = list(h5py.File(compressed_path, 'r').keys())  # read keys to access different stored frames
    original_n, final_n = 0, 0
    pre_filter_size, post_filter_size = 0, 0
    for i, slide_id in enumerate(tqdm(slide_ids, desc=f"Filtering ...")):
        slide_id_ = slide_id[:-1]  # FIXME ids in features file have a final undesired period
        roi_contours = read_annotations(args.data_dir, (slide_id,),
                                        annotation_dirname='tumour_area_annotations')[slide_id_]['Tumour area']
        area_contour = max((contour for contour in roi_contours if contour.shape[0] > 1 and contour.ndim == 3),
                           key=cv2.contourArea)
        # load contour from saved feature data
        data_path = next(path for path in (feature_dir/'data').iterdir() if path.name.startswith('data_' + slide_id))
        with data_path.open('r') as data_file:
            data = json.load(data_file)
        bounding_boxes_to_discard = []
        for item in data:
            contour = np.array(item['contour'])[:, np.newaxis, :]
            if not check_point_overlap(area_contour, contour):
                bounding_boxes_to_discard.append('{}_{}_{}_{}'.format(*item['bounding_rect']))
        bounding_boxes_to_discard = set(bounding_boxes_to_discard)  # make into set for easy containment look-up
        slide_features = pd.read_hdf(compressed_path, slide_id)
        pre_filter_size += slide_features.memory_usage().sum()
        original_n += len(slide_features)
        slide_features = slide_features[slide_features.index.map(lambda bb_str: bb_str not in bounding_boxes_to_discard)]  # YOU HAD MADE A CATASTROPHIC MISTAKE HERE, BY FORGETTING THE 'NOT'
        if len(slide_features) == 0:
            continue
        # additional subsampling to keep dataset small
        if args.subsample_fraction != 1.0:
            # estimate data distribution on dim reduced version of data and attempt to extract samples that follow the distribution
            dim_reduced_features = PCA(min(20, *slide_features.shape)).fit_transform(StandardScaler().fit_transform(slide_features))
            scores = np.exp(KernelDensity().fit(dim_reduced_features).score_samples(dim_reduced_features))
            weights = scores/scores.sum()
            indices = np.random.choice(np.arange(len(slide_features)),
                                       size=int(np.ceil(len(slide_features)*args.subsample_fraction)),
                                       p=weights, replace=False)
            slide_features = slide_features.iloc[indices]
        post_filter_size += slide_features.memory_usage().sum()
        final_n += len(slide_features)
        slide_features.to_hdf(filtered_path, key=slide_id)
    print(f"Done! {original_n - final_n} contours were filtered out of feature file ({bytes2human(pre_filter_size)} --> {bytes2human(post_filter_size)}")
    if args.subsample_fraction != 1.0:
        print(f"Only {args.subsample_fraction*100}% of the dataset was kept")


def select_label(args):  # TODO test
    feature_dir = args.data_dir/'data'/'features'/args.experiment_name
    compressed_path = feature_dir/'compressed.h5'
    labelled_path = feature_dir/'labelled.h5'
    slide_ids = list(h5py.File(compressed_path, 'r').keys())  # read keys to access different stored frames
    labels = pd.read_csv(args.skip_labels)
    original_n, final_n = 0, 0
    pre_filter_size, post_filter_size = 0, 0
    for i, slide_id in enumerate(tqdm(slide_ids, desc=f"Selecting datapoints by label ...")):
        slide_id_ = slide_id[:-1]  # FIXME ids in features file have a final undesired period
        slide_features = pd.read_hdf(compressed_path, slide_id)
        slide_labels = labels[labels['slide_id'] == slide_id_]
        box_ids = set([box_id for box_id, label in zip(slide_labels['box_id'], slide_labels['label'])
                       if label == args.label_num])
        pre_filter_size += slide_features.memory_usage().sum()
        original_n += len(slide_features)
        slide_features = slide_features[slide_features.index.map(lambda box_id: box_id in box_ids)]
        post_filter_size += slide_features.memory_usage().sum()
        final_n += len(slide_features)
        slide_features.to_hdf(labelled_path, key=slide_id)
    print(f"Done! {original_n - final_n} contours were filtered out of feature file ({bytes2human(pre_filter_size)} --> {bytes2human(post_filter_size)}")


# ########## subparser commands ############


def add_quantify_args(parser_quantify):
    parser_quantify.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser_quantify.add_argument('--experiment_name', type=str, required=True,
                                 help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    parser_quantify.add_argument('--outer_label', type=str, default='epithelium',
                                 help="Label whose instances are of interest for feature quantification")
    parser_quantify.add_argument('--reference_slide', type=str, default=None)
    parser_quantify.add_argument('--workers', type=int, default=4,
                                 help="Number of processes used in parallelized tasks")
    parser_quantify.add_argument('--shuffle_annotations', action='store_true',
                                 help="Shuffle processing order. Useful if multiple processing jobs are launched at the same time.")
    parser_quantify.add_argument('--overwrite', action='store_true', help="Whether to overwrite existing feature files")
    parser_quantify.add_argument('-ds', '--debug_slides', action='append', default=None)
    # parser_quantify.add_argument('--max_contour_num', type=int, default=3000,
    # help="Max number of glands in one slide to sample")


def add_compress_args(parser_compress):
    parser_compress.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser_compress.add_argument('--experiment_name', type=str, required=True,
                                 help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")


def add_dim_reduce_args(parser_dim_reduce):
    # not used
    parser_dim_reduce.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser_dim_reduce.add_argument('--n_components', type=int, default=20, help="Number of principal components in incremental PCA")
    parser_dim_reduce.add_argument('--overwrite_model', action='store_true')
    parser_dim_reduce.add_argument('--experiment_name', type=str,
                                   help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")


def add_filter_args(parser_filter):
    parser_filter.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser_filter.add_argument('--experiment_name', type=str, required=True,
                               help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    # additional subsampling
    parser_filter.add_argument('--subsample_fraction', type=float, default=1.0, help="Subsample the dataset further to keep only the important pieces")


def add_select_label_args(parser_single_key):
    parser_single_key.add_argument('data_dir', type=Path)
    parser_single_key.add_argument('experiment_name', type=str)
    parser_single_key.add_argument('label_file', type=Path)
    parser_single_key.add_argument('--label_num', type=int, default=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='task', help="Name of task to perform")
    add_quantify_args(subparsers.add_parser('quantify'))
    add_compress_args(subparsers.add_parser('compress'))
    add_dim_reduce_args(subparsers.add_parser('dim_reduce'))
    add_filter_args(subparsers.add_parser('filter'))
    add_select_label_args(subparsers.add_parser('select_label'))
    # general arguments
    args = parser.parse_args()
    if args.task is None:
        parser.print_help()
        sys.exit("No task name was given.")
    if args.task == 'quantify':
        quantify(args)
    elif args.task == 'compress':
        compress(args)
    elif args.task == 'dim_reduce':
        dim_reduce(args)
    elif args.task == 'filter':
        filter_(args)
    elif args.task == 'select_label':
        select_label(args)
    else:
        raise ValueError(f"Unknown task type {args.task}.")
