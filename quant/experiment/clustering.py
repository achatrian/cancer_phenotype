from pathlib import Path
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from skimage import transform, color, util
import cv2
import imageio
from scipy.spatial.distance import cdist
from . import BaseExperiment
from data.images.wsi_reader import WSIReader


def get_examples_per_subset(features, cluster_assignment, image_dir, n_examples=5, mpp=0.2, cluster_centers=None):
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
    for i, cluster in enumerate(tqdm(clusters, desc='clusters')):
        x_cluster = features.iloc[(cluster_assignment == cluster).to_numpy().squeeze()]
        examples.append(dict())
        for subset_id in tqdm(features.index.levels[0], desc='subsets'):
            try:
                subset_path = next((image_dir / (subset_id + sfx)) for sfx in ['.ndpi', '.svs', '.dzi'])
            except StopIteration:
                raise ValueError(f"Data dir does not contain images for {subset_id}")
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
                            0])  # if there are more examples than cluster protoypes cycle through them
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


def make_subset_examples_grid(save_dir, examples, experiment_name, image_size=512):
    r""""""
    save_dir = Path(save_dir)
    n_clusters = len(examples)
    n_subsets = len(examples[0])
    n_examples = len(next(iter(examples[0].values())))
    print(f"n subsets, clusters: {n_subsets}, {n_clusters}")
    for n in tqdm(range(n_examples)):
        grid = np.zeros((image_size * n_subsets, image_size * n_clusters, 3))
        if n == 0:
            print(f"Grid size: {grid.size}")
        for j, cluster_examples in enumerate(examples):
            for i, (subset_id, subset_examples) in enumerate(cluster_examples.items()):
                if len(subset_examples) >= n+1:
                    example = np.array(subset_examples[n])
                    max_dim = max(example.shape[:2])
                    padded = np.pad(example, (
                            (0, max(0, max_dim - example.shape[0])),
                            (0, max(0, max_dim - example.shape[1])),
                            (0, 0)
                        ), 'constant')
                    resized = transform.resize(padded, output_shape=(image_size,)*2)
                    resized = (resized * 255.0).astype(np.uint8)
                    grid[i*image_size:(i+1)*image_size, j*image_size:(j+1)*image_size] = resized
                    # TODO replace below with PIL utilities such as ImageFont and ImageDraw
                    cv2.putText(grid, f'{j}', (j*image_size, i*image_size), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,)*3)
                else:
                    print(f"No examples for cluster {j} in slide {subset_id}")
                    cv2.putText(grid, 'NA', (j*image_size, i*image_size), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,)*3)
        if n_subsets > n_clusters * 2:
            first_half, second_half = grid[:image_size*n_subsets//2], grid[image_size*n_subsets//2:]
            grid = np.concatenate((first_half, second_half), axis=1)
        if n_clusters > n_subsets * 2:
            first_half, second_half = grid[:, :image_size*n_clusters//2], grid[:, image_size*n_clusters//2:]
            grid = np.concatenate((first_half, second_half), axis=1)
        imageio.imwrite(save_dir/f'grid_{experiment_name}{n}.png', grid)
    with open(save_dir/'details.json', 'w') as details_file:
        json.dump({
            'experiment_name': experiment_name,
            'files': list(examples[0].keys()),
            'type': 'cluster_subset_grid'
        }, details_file)


def make_cluster_grids(examples, save_dir, experiment_name, image_size=512):
    r""""""
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
            cluster_grid = util.montage(np.array(cluster_images))
            imageio.imwrite(cluster_dirs[n]/f'cluster_{c}_example{n}.png', cluster_grid)
    with open(cluster_dirs/'details.json', 'w') as details_file:
        json.dump({
            'experiment_name': experiment_name,
            'files': list(examples[0].keys()),
            'type': 'cluster_subset_grid'
        }, details_file)
