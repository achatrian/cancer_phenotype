from pathlib import Path
import json
import warnings
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from skimage import transform, color
import cv2
import imageio
from scipy.spatial.distance import cdist
from experiment import BaseExperiment
from data.images.wsi_reader import make_wsi_reader, add_reader_args, get_reader_options

# TODO adapt to BaseExperiment


class Clustering(BaseExperiment):
    def __init__(self, args):
        super().__init__(args)

    def plot_clusters(self, colors=('b', 'g', 'r', 'c', 'm'), dim_reduction='PCA'):
        r"""Plot clusters for whole dataset. Assume y is vector of cluster indexes"""
        if self.y is None:
            raise ValueError("No clustering data available (call clustering.run())")
        self.clusters = np.unique(self.y)
        if len(colors) < len(self.clusters):
            raise ValueError(f"Not enough colors to plot {len(self.clusters)} different clusters")
        # if x is too large, shrink to 2D for plotting
        x = self.x if len(self.x.columns) == 2 else self.dim_reduction(ndim=2, type=dim_reduction)
        for i, cluster in enumerate(self.clusters):
            ix = np.where(cluster == self.y)[0]
            plt.scatter(x.iloc[ix, 0], x.iloc[ix, 1], color=colors[i])

    def get_examples(self, image_dir=None, n_examples=5, mpp=0.2, cluster_centers=None, image_dim_increase=0.5):
        r"""Extract cluster examples directly from images"""
        if self.x is None or self.y is None:
            raise ValueError("Data has not been read / processed yet for cluster extraction")
        if image_dir is None:
            image_dir = Path(self.loaded_paths[0]).parents[2]
            assert (image_dir / 'data').is_dir(), "image_dir should contain dir 'data'"
        if self.clusters is None:
            self.clusters = np.unique(self.y)
        examples = []
        image_dir = Path(image_dir)
        image_paths = tuple(image_dir.glob('*.ndpi')) + tuple(image_dir.glob('*.svs')) + tuple(image_dir.glob('*.dzi'))\
            + tuple(image_dir.glob('*/*.ndpi')) + tuple(image_dir.glob('*/*.svs')) + tuple(image_dir.glob('*/*.dzi'))
        for i, cluster in enumerate(tqdm(self.clusters, desc='clusters')):
            x_cluster = self.x.iloc[(self.y == cluster).to_numpy().squeeze()]
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
                elif len(cluster_centers) == len(self.clusters) and cluster_centers[0].ndim == 2:
                    # case where cluster is described by multiple points (e.g. som + mc)
                    cc = cluster_centers[i]  #
                    dist = cdist(cc, x_cluster.to_numpy(), 'euclidean').squeeze()
                    for n in range(n_examples):
                        nn = n % min(n_examples, dist.shape[
                            0])  # if there are more examples than cluster protoypes cycle through them
                        closest_points_indices.append(int(dist[nn, ...].argmin()))
                        dist[:, closest_points_indices[-1]] = np.inf  # make column corresponding to example nn infinite
                else:
                    raise ValueError("Invalid format for cluster_centers parameter")
                if np.unique(closest_points_indices).size != len(closest_points_indices):
                    warnings.warn(f"Only {np.unique(closest_points_indices).size} unique examples were found for cluster '{cluster}'")
                sample_index = tuple(x_cluster.iloc[closest_points_indices].index)
            for subset_id, bb_s in sample_index:
                x, y, w, h = tuple(int(d) for d in bb_s.split('_'))
                x -= int(w * image_dim_increase/2)  # expand bounding box to give better view of gland
                y -= int(h * image_dim_increase/2)
                w += int(w * image_dim_increase)
                h += int(h * image_dim_increase)
                try:
                    subset_path = next(path for path in image_paths if subset_id == path.with_suffix('').name)
                except StopIteration:
                    raise FileNotFoundError(f"DataFrame key: {subset_id} does not match an image file")
                reader = make_wsi_reader(file_name=str(subset_path))
                image = np.array(reader.read_region((x, y), 0, (w, h)))  # changed level from None to 0 !!!
                if image.shape[2] == 4:  # assume 4 channels images are RGBA
                    image = color.rgba2rgb(image)
                if image.max() <= 1.0 and image.min() >= 0.0:
                    image = image * 255.0
                image = image.astype(np.uint8)
                examples[i].append(image)
        return examples

    def get_examples_per_subset(self, image_dir=None, n_examples=5, mpp=0.2, cluster_centers=None):
        r"""This methods assumes that the dataframe's index corresponds to the bounding box of tissue elements
        :param image_dir
        :param n_examples: how many images examples to extract for images per cluster
        :param mpp:
        :param cluster_centers: if not empty, examples that are closest to the cluster centers are taken
        """
        if self.x is None or self.y is None:
            raise ValueError("Data has not been read / processed yet for cluster extraction")
        if image_dir is None:
            image_dir = Path(self.loaded_paths[0]).parents[2]
            assert (image_dir / 'data').is_dir(), "image_dir should contain dir 'data'"
        if self.clusters is None:
            self.clusters = np.unique(self.y)
        examples = []
        for i, cluster in enumerate(tqdm(self.clusters, desc='clusters')):
            x_cluster = self.x.iloc[(self.y == cluster).to_numpy().squeeze()]
            examples.append(dict())
            for subset_id in tqdm(self.x.index.levels[0], desc='subsets'):
                try:
                    subset_path = next((image_dir / (subset_id + sfx)) for sfx in ['.ndpi', '.svs', '.dzi'])
                except StopIteration:
                    raise ValueError(f"Data dir does not contain images for {subset_id}")
                examples[i][subset_id] = []
                opt = get_reader_options(include_path=False)
                reader = make_wsi_reader(subset_path, opt)
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
                    elif len(cluster_centers) == len(self.clusters) and cluster_centers[0].ndim == 2:
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

    def save_examples_grid(self, save_dir, examples, image_size=256):
        r""""""
        save_dir = Path(save_dir)
        n_clusters = len(examples)
        n_examples = len(examples[0])
        print(f"n clusters: {n_clusters}")
        grid = np.zeros((image_size * n_examples, image_size * n_clusters, 3))
        for n in tqdm(range(n_examples)):
            if n == 0:
                print(f"Grid size: {grid.size}")
            for j, cluster_examples in enumerate(examples):
                example = np.array(cluster_examples[n])
                max_dim = max(example.shape[:2])
                padded = np.pad(example, (
                    (0, max(0, max_dim - example.shape[0])),
                    (0, max(0, max_dim - example.shape[1])),
                    (0, 0)
                ), 'constant')
                resized = transform.resize(padded, output_shape=(image_size,) * 2)
                resized = (resized * 255.0).astype(np.uint8)
                grid[n * image_size:(n + 1) * image_size, j * image_size:(j + 1) * image_size] = resized
                # TODO replace below with PIL utilities such as ImageFont and ImageDraw
                cv2.putText(grid, f'{j}', (j * image_size, n * image_size), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255,) * 3)
        imageio.imwrite(save_dir / f'examples_grid_{self.name}.png', grid)
        with open(save_dir / 'details.json', 'w') as details_file:
            json.dump({
                'experiment_name': self.name,
                'type': 'cluster_grid'
            }, details_file)
        print("Done!")

    def save_subset_examples_grid(self, save_dir, examples, image_size=512):
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
            imageio.imwrite(save_dir/f'grid_{self.name}{n}.png', grid)
        with open(save_dir/'details.json', 'w') as details_file:
            json.dump({
                'experiment_name': self.name,
                'files': list(examples[0].keys()),
                'type': 'cluster_subset_grid'
            }, details_file)
        print("Done!")













