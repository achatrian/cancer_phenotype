import json
import warnings
from pathlib import Path
from random import shuffle

import imageio
import numpy as np
from scipy.spatial.distance import cdist
from skimage import color, transform, util
from tqdm import tqdm

from wsi_reader import WSIReader


def get_cluster_examples(features, cluster_assignment, image_dir, n_examples=5, mpp=0.2, cluster_centers=None, clusters=None, image_dim_increase=0.5):
    r"""Extract cluster examples directly from images"""
    assert (image_dir / 'data').is_dir(), "image_dir should contain dir 'data'"
    if clusters is None:
        clusters = np.unique(cluster_assignment)
    examples = []
    image_dir = Path(image_dir)
    image_paths = tuple(image_dir.glob('*.ndpi')) + tuple(image_dir.glob('*.svs')) + tuple(image_dir.glob('*.dzi')) \
                  + tuple(image_dir.glob('*/*.ndpi')) + tuple(image_dir.glob('*/*.svs')) + tuple(
        image_dir.glob('*/*.dzi'))
    for i, cluster in enumerate(tqdm(clusters, desc='clusters')):
        x_cluster = features.iloc[(cluster_assignment == cluster).to_numpy().squeeze()]
        examples.append([])
        if cluster_centers is None:
            sample_index = tuple(
                x_cluster.sample(n=n_examples, replace=False).index)  # randomly sample examples
            if len(sample_index) < n_examples:
                sample_index = sample_index + sample_index[:n_examples]
        else:
            closest_points_indices = []  # list of lists, each list containing the index of a close point to one cluster
            if isinstance(cluster_centers, np.ndarray) and cluster_centers.ndim == 2:
                dist = cdist(cluster_centers[i:i + 1, ...], x_cluster.to_numpy(), 'euclidean').squeeze()
                for n in range(n_examples):
                    closest_points_indices.append(int(dist.argmin()))
                    dist[closest_points_indices[-1]] = np.inf
            elif len(cluster_centers) == len(clusters) and cluster_centers[0].ndim == 2:
                # case where cluster is described by multiple points (e.g. som + mc)
                cc = cluster_centers[i].copy()
                shuffle(cc)  # shuffle to get examples from cluster representatives that may be far away
                dist = cdist(cc, x_cluster.to_numpy(), 'euclidean').squeeze()
                for n in range(n_examples):
                    nn = n % min(n_examples, dist.shape[0])  # if there are more examples than cluster protoypes cycle through them
                    closest_points_indices.append(int(dist[nn, ...].argmin()))
                    dist[:, closest_points_indices[-1]] = np.inf  # make column corresponding to example nn infinite
            else:
                raise ValueError("Invalid format for cluster_centers parameter")
            if np.unique(closest_points_indices).size != len(closest_points_indices):
                warnings.warn(
                    f"Only {np.unique(closest_points_indices).size} unique examples were found for cluster '{cluster}'")
            sample_index = tuple(x_cluster.iloc[closest_points_indices].index)
        for subset_id, bb_s in sample_index:
            x, y, w, h = tuple(int(d) for d in bb_s.split('_'))
            x -= int(w * image_dim_increase / 2)  # expand bounding box to give better view of gland
            y -= int(h * image_dim_increase / 2)
            w += int(w * image_dim_increase)
            h += int(h * image_dim_increase)
            try:
                subset_path = next(path for path in image_paths if subset_id == path.with_suffix('').name)
            except StopIteration:
                raise FileNotFoundError(f"DataFrame key: {subset_id} does not match an image file")
            reader = WSIReader(file_name=str(subset_path))
            image = np.array(reader.read_region((x, y), 0, (w, h)))  # changed level from None to 0 !!!
            if image.shape[2] == 4:  # assume 4 channels images are RGBA
                image = color.rgba2rgb(image)
            if image.max() <= 1.0 and image.min() >= 0.0:
                image = image * 255.0
            image = image.astype(np.uint8)
            examples[i].append(image)
    return examples


def get_cluster_examples_per_subset(features, cluster_assignment, image_dir, n_examples=5, mpp=0.2, cluster_centers=None):
    r"""assumes that the dataframe's index corresponds to the bounding box of tissue elements
    :param image_dir
    :param n_examples: how many images examples to extract for images per cluster
    :param mpp:
    :param cluster_centers: if not empty, examples that are closest to the cluster centers are taken
    :returns examples: the nth entry corresponds to cluster n, and it's a dictionary with one key per image/subset
            Its values are lists of example component images for each subset for the nth cluster
    """
    if features is None or cluster_assignment is None:
        raise ValueError("Data has not been read / processed yet for cluster extraction")
    clusters = np.unique(cluster_assignment)
    examples = []
    image_paths = tuple(image_dir.glob('*.ndpi')) + tuple(image_dir.glob('*.svs')) + tuple(image_dir.glob('*.dzi')) \
                  + tuple(image_dir.glob('*/*.ndpi')) + tuple(image_dir.glob('*/*.svs')) + tuple(image_dir.glob('*/*.dzi'))
    for i, cluster in enumerate(tqdm(clusters, desc='clusters')):
        x_cluster = features.iloc[(cluster_assignment == cluster).to_numpy().squeeze()]
        examples.append(dict())
        for subset_id in tqdm(features.index.levels[0], desc='subsets'):
            try:
                subset_path = next(path for path in image_paths if subset_id == path.with_suffix('').name)
            except StopIteration:
                raise FileNotFoundError(f"DataFrame key: {subset_id} does not match an image file")
            examples[i][subset_id] = []
            opt = WSIReader.get_reader_options(include_path=False)
            reader = WSIReader(subset_path, opt)
            try:
                x_subset = x_cluster.loc[subset_id]
            except KeyError:
                continue
            if cluster_centers is None:
                sample_index = tuple(
                    x_subset.sample(n=n_examples, replace=False).index)  # randomly sample examples
                if len(sample_index) < n_examples:
                    sample_index = sample_index + sample_index[:n_examples]
            else:
                closest_points_indices = []  # list of lists, each list containing the index of a close point to one cluster
                if isinstance(cluster_centers, np.ndarray) and cluster_centers.ndim == 2:
                    dist = cdist(cluster_centers[i:i + 1, ...], x_cluster.to_numpy(), 'euclidean').squeeze()
                    for n in range(n_examples):
                        closest_points_indices.append(int(dist.argmin()))
                        dist[closest_points_indices[-1]] = np.inf
                elif len(cluster_centers) == len(clusters) and cluster_centers[0].ndim == 2:
                    # case where cluster is described by multiple points (e.g. som + mc)
                    cc = cluster_centers[i]  #
                    dist = cdist(cc, x_cluster.to_numpy(), 'euclidean').squeeze()
                    for n in range(n_examples):
                        nn = n % min(n_examples, dist.shape[
                            0])  # if there are more examples than cluster prototypes cycle through them
                        closest_points_indices.append(int(dist[nn, ...].argmin()))
                        dist[:, closest_points_indices[-1]] = np.inf  # make column corresponding to example nn infinite
                if np.unique(closest_points_indices).size != len(closest_points_indices):
                    warnings.warn(f"Only {np.unique(closest_points_indices).size} unique examples were found for cluster '{cluster}' subset '{subset_id}'")
                sample_index = tuple(x_subset.iloc[closest_points_indices].index)
            for bb_s in sample_index:
                x, y, w, h = tuple(int(d) for d in bb_s.split('_'))
                image = np.array(reader.read_region((x, y), 0, (w, h)))  # changed level from None to 0 !!!
                if image.shape[2] == 4:  # assume 4 channels images are RGBA
                    image = color.rgba2rgb(image)
                if image.max() <= 1.0 and image.min() >= 0.0:
                    image = image * 255.0
                image = image.astype(np.uint8)
                examples[i][subset_id].append(image)
    return examples


def make_cluster_grids(examples, save_dir, experiment_name, image_size=512):
    r""""""
    save_dir = Path(save_dir, 'clusters_grids')
    save_dir.mkdir(exist_ok=True, parents=True)
    n_clusters = len(examples)
    n_examples = len(examples[0])
    print(f"n clusters: {n_clusters}")
    for c, cluster_examples in enumerate(examples):
        cluster_images = []
        for n in tqdm(range(n_examples)):
            example = np.array(cluster_examples[n])
            max_dim = max(example.shape[:2])
            padded = np.pad(example, (
                (0, max(0, max_dim - example.shape[0])),
                (0, max(0, max_dim - example.shape[1])),
                (0, 0)
            ), 'constant')
            resized = transform.resize(padded, output_shape=(image_size,) * 2)
            resized = (resized * 255.0).astype(np.uint8)
            cluster_images.append(resized)
        cluster_grid = util.montage(np.array(cluster_images), multichannel=True)
        imageio.imwrite(save_dir / f'cluster{c}.png', cluster_grid)
    with open(save_dir / 'details.json', 'w') as details_file:
        json.dump({
            'experiment_name': experiment_name,
            'type': 'cluster_grid'
        }, details_file)
    print("Done!")


def make_cluster_grids_per_subset(examples, save_dir, experiment_name, image_size=512):
    r"""Multiple grid showing cluster membership per each slide"""
    save_dir = Path(save_dir)
    n_clusters = len(examples)
    n_subsets = len(examples[0])
    n_examples = len(next(iter(examples[0].values())))
    grid_ver_n_images = np.ceil(np.sqrt(n_subsets))
    grid_hor_n_images = n_subsets//grid_ver_n_images
    print(f"n subsets, clusters: {n_subsets}, {n_clusters}")
    clusters_dir = Path(save_dir, f'{experiment_name}_cluster_grids')
    clusters_dir.mkdir(exist_ok=True, parents=True)
    cluster_dirs = []
    for i in range(n_clusters):
        cluster_dirs.append(Path(clusters_dir, f'{i}'))
        cluster_dirs[-1].mkdir(exist_ok=True, parents=True)
    for n in tqdm(range(n_examples)):
        for c, cluster_examples in enumerate(examples):
            cluster_images = []
            for subset_id, subset_examples in cluster_examples.items():
                try:
                    example = np.array(subset_examples[n])
                except IndexError:
                    break
                max_dim = max(example.shape[:2])
                padded = np.pad(example, (
                        (0, max(0, max_dim - example.shape[0])),
                        (0, max(0, max_dim - example.shape[1])),
                        (0, 0)
                    ), 'constant')
                resized = transform.resize(padded, output_shape=(image_size,)*2)
                resized = (resized * 255.0).astype(np.uint8)
                cluster_images.append(resized)
            cluster_grid = util.montage(np.array(cluster_images), multichannel=True)
            imageio.imwrite(cluster_dirs[n]/f'cluster_{c}_example{n}.png', cluster_grid)
    with open(clusters_dir/'details.json', 'w') as details_file:
        json.dump({
            'experiment_name': experiment_name,
            'files': list(examples[0].keys()),
            'type': 'cluster_subset_grid'
        }, details_file)