from pathlib import Path
from argparse import ArgumentParser
import json
from functools import reduce
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import OPTICS
from imageio import imwrite
from base.utils.utils import pairwise
from data.images.wsi_reader import make_wsi_reader
from quant.viz import make_cluster_grids


r"""
Find ratio of neighbors that belong to biopsies vs RPs for each gland.
Use ratio values to obtain examples that are typical of biopsies or RPs
"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, default='/well/rittscher/projects/ProMPT/cases')
    parser.add_argument('--save_dir', type=Path, default='/well/rittscher/projects/ProMPT/data')
    parser.add_argument('--recursive_search', action='store_true')
    parser.add_argument('--experiment_name', type=str, default='combined_mpp1.0_normal_nuclei_circles')
    parser.add_argument('--image_suffix', type=str, default=['isyntax'], action='append', choices=['tiff', 'svs', 'ndpi', 'isyntax'], help="process images with these extension")
    parser.add_argument('--num_neighbors', type=int, default=50)
    parser.add_argument('--nr_save_path', type=Path, default='/well/rittscher/projects/ProMPT/data/results/neighbour_biopsy_rp.csv')
    parser.add_argument('--num_cluster_examples', type=int, default=9)
    args = parser.parse_args()

    features_dir = args.save_dir/'features'/args.experiment_name
    features_paths = list(features_dir.glob('*.json'))
    with open('/well/rittscher/projects/ProMPT/cases/pairs.json', 'r') as pairs_file:
        pairs = json.load(pairs_file)
    pairs = [p['cases'] for p in pairs]
    biopsies, rps = set(p[0] for p in pairs), set(p[1] for p in pairs)
    features, slides_ids = [], []
    b_rp_labels = []
    for feature_path in tqdm(features_paths, desc='reading features ...'):
        slide_id = feature_path.with_suffix('').name
        f = pd.read_json(feature_path, orient='split', convert_axes=False)  # default behaviour reads bb string as date
        if int(slide_id.split('_')[0]) in biopsies:
            b_rp_labels.extend([0]*len(f))
        elif int(slide_id.split('_')[0]) in rps:
            b_rp_labels.extend([1]*len(f))
        else:
            continue
        features.append(f)
        slides_ids.extend([slide_id]*len(f))  # use to know which slide gland is from
    features = pd.concat(features)
    features.index = pd.MultiIndex.from_arrays((slides_ids, features.index), names=('slide_id', 'bounding_box'))
    try:
        neighbors_ratio = nr = pd.read_csv(args.nr_save_path, index_col=[0, 1])
    except FileNotFoundError:
        scaler = StandardScaler().fit(features)
        nn = NearestNeighbors(args.num_neighbors).fit(scaler.transform(features))
        #rp_neighbour_ratio, distances_from_biopsy, distances_from_rp = [], [], []
        def pyf(e): return b_rp_labels[e]
        pyf = np.vectorize(pyf)
        print("Finding neighbors ...")
        distances, indices = nn.kneighbors(scaler.transform(features))
        labels = pyf(indices)
        print("Computing distances and ratio ...")
        biopsy_distances = distances.copy()
        biopsy_distances[labels] = np.nan  # nan rp entries
        distances_from_biopsy = np.nanmedian(biopsy_distances, axis=1)
        rp_distances = distances.copy()
        rp_distances[np.logical_not(labels)] = np.nan  # nan biopsy entries
        distances_from_rp = np.nanmedian(rp_distances, axis=1)
        rp_neighbor_ratio = labels.mean(axis=1)
        neighbors_ratio = nr = pd.DataFrame({
            'rp_neighbor_ratio': rp_neighbor_ratio,
            'distance_from_biopsy': distances_from_biopsy,
            'distances_from_rp': distances_from_rp,
            'biopsy_rp_label': b_rp_labels
        }, index=features.index)
        neighbors_ratio.to_csv(args.nr_save_path)
    # obtain examples from ratio classes
    histogram, bin_edges = np.histogram(neighbors_ratio['rp_neighbor_ratio'], bins=np.arange(0, 1, 0.1))
    image_paths = []
    save_dir = Path(args.save_dir) if args.save_dir is not None else Path(args.data_dir)/'data'
    for suffix in args.image_suffix:
        image_paths.extend(args.data_dir.glob(f'./*.{suffix}'))
        if args.recursive_search:
            image_paths.extend(args.data_dir.glob(f'*/*.{suffix}'))
    min_area, image_dim_increase = 200**2, 0.5
    for lower, upper in tqdm(pairwise(bin_edges), total=len(bin_edges) - 1):
        nr_bin = nr[(nr['rp_neighbor_ratio'] >= lower) & (nr['rp_neighbor_ratio'] < upper)]
        bounding_box_areas = np.array([reduce(lambda p, fa: p*fa, tuple(int(i) for i in bounding_box.split('_'))[2:])
                                       for bounding_box in nr_bin.index.get_level_values('bounding_box')])
        nr_bin = nr_bin[bounding_box_areas > min_area]
        bin_features = features.loc[nr_bin.index.unique()]
        cluster_labels = pd.Series(OPTICS().fit_predict(bin_features), index=nr_bin.index)
        bin_save_dir = save_dir/'results'/f'examples_neighbor_ratio_{lower}_{upper}'
        bin_save_dir.mkdir(exist_ok=True, parents=True)
        tqdm.write(f"{len(cluster_labels.unique())} clusters for bin [{lower}, {upper})")
        clusters = cluster_labels.unique()
        examples, examples_bbs = [], []
        for c in tqdm(clusters, desc='clusters'):
            if c == -1:  # outliers
                continue
            bin_examples = nr_bin[cluster_labels == c]
            bin_examples = bin_examples.sample(n=args.num_cluster_examples, replace=True)  # to create grid need exact num of images
            previous_slide_id, slide = '', None
            cluster_examples = []
            for (slide_id, bounding_box), row in bin_examples.iterrows():
                if previous_slide_id != slide_id:
                    image_path = next(path for path in image_paths if path.with_suffix('').name == slide_id)
                    slide = make_wsi_reader(image_path)
                x, y, w, h = tuple(int(i) for i in bounding_box.split('_'))
                x -= int(w * image_dim_increase / 2)  # expand bounding box to give better view of gland
                y -= int(h * image_dim_increase / 2)
                w += int(w * image_dim_increase)
                h += int(h * image_dim_increase)
                example = slide.read_region((x, y), 0, (w, h))
                cluster_examples.append(example)
                previous_slide_id = slide_id
            examples.append(cluster_examples)
            examples_bbs.append(bin_examples.index.get_level_values('bounding_box').tolist())
        make_cluster_grids(examples, bin_save_dir, f'bin_{lower}_{upper}', details=examples_bbs)
    print("Done!")





