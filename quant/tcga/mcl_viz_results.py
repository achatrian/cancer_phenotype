import argparse
from pathlib import Path
import re
import json
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from quant.utils import to_num


def make_grids_and_plot_heatmap(evaluation_results, x_param, y_param, z_param, ax=None, fig=None, **fixed_parameters):
    x_param_values = sorted(set(er[x_param] for er in evaluation_results.values()))
    y_param_values = sorted(set(er[y_param] for er in evaluation_results.values()))
    x_param_grid, y_param_grid = np.meshgrid(x_param_values, y_param_values)
    # x_param_values / inflation vs modularity and
    for er in evaluation_results.values():
        if 'silouhette_score' in er:
            er['silhouette_score'] = er['silouhette_score']
        if 'silhouette_score' in er:
            er['silouhette_score'] = er['silhouette_score']
    z_param_grid = np.full_like(x_param_grid, np.nan, dtype=np.float)  # np.nan needs floats array
    for n, (x_param_value, y_param_value) in enumerate(zip(x_param_grid.flatten(), y_param_grid.flatten())):
        y, x = np.unravel_index(n, z_param_grid.shape)  # get linear index through grid
        try:
            z_param_grid[y, x] = next(er[z_param] for er in evaluation_results.values()
                                      if er[x_param] == x_param_value and er[y_param] == y_param_value
                                      and all(er[k] == v for k, v in fixed_parameters.items()))
        except StopIteration:
            z_param_grid[y, x] = np.nan
    z_param_grid[z_param_grid == np.nan] = z_param_grid.mean()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x_param_grid, y_param_grid, z_param_grid, cmap=cm.coolwarm,
    #                 linewidth=0, antialiased=False)
    # ax.set_xlabel(x_param), ax.set_ylabel(y_param), ax.set_zlabel(z_param)
    # heatmap
    if ax is None:
        fig = plt.figure()
        im = plt.imshow(z_param_grid, cmap=cm.coolwarm)
        ax = plt.gca()
        fig.colorbar(im, ax=ax, orientation='horizontal' if len(x_param_values) > len(y_param_values) else 'vertical')
    else:
        if fig is None:
            raise ValueError("If 'ax' is not None, 'fig' must not be None")
        im = ax.imshow(z_param_grid, cmap=cm.coolwarm)
        fig.colorbar(im, ax=ax, orientation='horizontal' if len(x_param_values) > len(y_param_values) else 'vertical')
    ax.set_xlabel(x_param), ax.set_ylabel(y_param), ax.set_title(z_param)
    ax.set_xticks(np.arange(len(x_param_values))), ax.set_yticks(np.arange(len(y_param_values)))
    ax.set_xticklabels(x_param_values), ax.set_yticklabels(y_param_values)
    ax.set_title(f'{z_param} - ' + ''.join([f'{k}={v}' for k, v in fixed_parameters.items()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=Path, help="Directory where data is stored")
    parser.add_argument('--experiment', type=str, help="Experiment type - used to load desired experiment class")
    args = parser.parse_args()
    results_dir = args.data_dir / 'data' / 'experiments' / f'{args.experiment}' / 'results'
    visual_result_dir = args.data_dir / 'data' / 'experiments' / f'{args.experiment}' / 'results_viz'
    visual_result_dir.mkdir(exist_ok=True, parents=True)
    evaluation_results = {}
    # incorporate parameters into result
    for result_path in results_dir.iterdir():
        if not result_path.is_dir():
            continue
        parameters_values = [to_num(v) for v in re.findall('\d+\.?\d*', result_path.name)]  # REGEX ONLY WORKS FOR NUMERIC PARAMETER VALUES
        parameter_names = re.sub(':\d+\.?\d*', ' ', result_path.name).split(' ')[:-1]
        assert len(parameter_names) == len(parameters_values), "Equal number of names and values"
        parameters = {n: v for n, v in zip(parameter_names, parameters_values)}
        try:
            with open(result_path/'evaluation_results.json', 'r') as evaluation_file:
               evaluation_result = json.load(evaluation_file)
        except FileNotFoundError:
            continue
        evaluation_result.update(parameters)
        evaluation_results[result_path.name] = evaluation_result  #
    # num neighbours and inflation vs modularity
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    for i, map_size in enumerate([80, 90, 100, 110]):
        make_grids_and_plot_heatmap(evaluation_results, 'num_neighbors', 'inflation', 'modularity',
                                    ax=axes[i], fig=fig, map_size=map_size)
    fig.savefig(visual_result_dir / f'num_neighbors&inflation_vs_modularity.png')
    # num neighbours and inflation vs silhouette score
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    for i, map_size in enumerate([80, 90, 100, 110]):
        make_grids_and_plot_heatmap(evaluation_results, 'num_neighbors', 'inflation', 'silhouette_score',
                                    ax=axes[i], fig=fig, map_size=map_size)
    fig.savefig(visual_result_dir / f'num_neighbors&inflation_vs_silhouette.png')
    # num neighbors and inflation vs num_clusters
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    for i, map_size in enumerate([80, 90, 100, 110]):
        make_grids_and_plot_heatmap(evaluation_results, 'num_neighbors', 'inflation', 'num_clusters',
                                    ax=axes[i], fig=fig, map_size=map_size)
    fig.savefig(visual_result_dir / f'num_neighbors&inflation_vs_num_clusters.png')
    # num neighbors and inflation vs adjusted mutual information
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    for i, map_size in enumerate([80, 90, 100, 110]):
        make_grids_and_plot_heatmap(evaluation_results, 'num_neighbors', 'inflation', 'adjusted_mutual_information_score',
                                    ax=axes[i], fig=fig, map_size=map_size)
    fig.savefig(visual_result_dir / f'num_neighbors&inflation_vs_adjusted_mutual_information_score.png')
    # num neighbors and inflation vs adjusted rand score
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    for i, map_size in enumerate([80, 90, 100, 110]):
        make_grids_and_plot_heatmap(evaluation_results, 'num_neighbors', 'inflation', 'adjusted_rand_score',
                                    ax=axes[i], fig=fig, map_size=map_size)
    fig.savefig(visual_result_dir / f'num_neighbors&inflation_vs_adjusted_rand_score.png')
    # num neighbors and inflation vs random forest gleason prediction
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    for i, map_size in enumerate([80, 90, 100, 110]):
        make_grids_and_plot_heatmap(evaluation_results, 'num_neighbors', 'inflation', 'rf_average_gleason_prediction_score',
                                    ax=axes[i], fig=fig, map_size=map_size)
    fig.savefig(visual_result_dir / f'num_neighbors&inflation_vs_rf_average_gleason_prediction_score.png')
    print("Done!")



