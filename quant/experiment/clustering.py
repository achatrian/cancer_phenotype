from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from skimage import transform, color
import cv2
import imageio
from . import Experiment
from base.data.wsi_reader import WSIReader


class Clustering(Experiment):
    def __init__(self, name, step_names, steps, outlier_removal=IsolationForest(), caching_path=None):
        super().__init__(name, step_names, steps, outlier_removal, caching_path=None)
        self.clusters = None

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

    def get_examples(self, data_dir=None, n_examples=5, mpp=0.2):
        r"""This methods assumes that the dataframe's index corresponds to the bounding box of tissue elements
        :param n_examples: how many image examples to extract for image per cluster
        :param mpp:
        """
        if self.x is None or self.y is None:
            raise ValueError("Data has not been read / processed yet for cluster extraction")
        if data_dir is None:
            data_dir = Path(self.loaded_paths[0]).parents[2]
            assert (data_dir / 'data').is_dir(), "data_dir should contain dir 'data'"
        self.clusters = np.unique(self.y)
        examples = []
        for i, cluster in enumerate(tqdm(self.clusters, desc='clusters')):
            x_cluster = self.x.iloc[(self.y == cluster).to_numpy().squeeze()]
            examples.append(dict())
            for subset_id in tqdm(self.x.index.levels[0], desc='subsets'):
                try:
                    subset_path = next((data_dir / (subset_id + sfx)) for sfx in ['.ndpi', '.svs', '.dzi'])
                except StopIteration:
                    raise ValueError(f"Data dir does not contain image for {subset_id}")
                examples[i][subset_id] = []
                opt = WSIReader.get_reader_options(include_path=False)
                reader = WSIReader(opt, subset_path)
                try:
                    x_subset = x_cluster.loc[subset_id]
                except KeyError:
                    continue
                sample = x_subset.sample(n=n_examples, replace=True)
                for bb_s, row in sample.iterrows():
                    x, y, w, h = tuple(int(d) for d in bb_s.split('_'))
                    image = np.array(reader.read_region((x, y), 0, (w, h)))  # changed level from None to 0 !!!
                    if image.shape[2] == 4:  # assume 4 channels images are RGBA
                        image = color.rgba2rgb(image)
                    if image.max() <= 1.0 and image.min() >= 0.0:
                        image = image * 255.0
                    image = image.astype(np.uint8)
                    examples[i][subset_id].append(image)
        return examples

    def save_examples_grid(self, save_dir, examples, image_size=512):
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
            imageio.imwrite(save_dir/f'grid{n}.png', grid)
        with open(save_dir/'details.json', 'w') as details:
            json.dump({
                'experiment_name': self.name,
                'files': list(examples[0].keys()),
            }, details)
        print("Done!")













