from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('checkpoints_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    args = parser.parse_args()
    results_file = Path(args.checkpoints_dir/'experiments'/args.experiment_name/'results.csv')
    save_dir = Path(args.checkpoints_dir/'experiments'/args.experiment_name/'results_plots')
    save_dir.mkdir(exist_ok=True)
    results = pd.read_csv(results_file, delimiter='\t')
    # prune epochs -- find streak with highest end number, and go back to its zero
    end_index = np.argmax(results['epoch'])
    start_index = np.where(results['epoch'].iloc[:end_index] == 0)[0][-1]
    results = results.iloc[start_index:end_index]
    results['epoch_iters'] = results['epoch'] + results['iters']/results['iters'].max()
    # gather together all metrics whose name has the same start (e.g. for multiclass accuracy)
    already_plotted = {'epoch', 'iters', 'epoch_iters'}
    plots_titles = set()
    for col in results:
        if col in already_plotted:
            continue
        filtered_col = ['epoch_iters']
        for col_ in results:
            if col_.startswith(col):
                filtered_col.append(col_)
                already_plotted.add(col)
                already_plotted.add(col_)
        filtered_results = results[filtered_col]
        title_col = min(filtered_col, key=lambda s: len(s))
        plots_titles.add(title_col)
        filtered_col.remove('epoch_iters')  # for selecting y traces
        filtered_results.plot(x='epoch_iters', legend=True, title=f'{args.experiment_name}_{title_col}')
        plt.savefig(save_dir/f'{args.experiment_name}_{title_col}.png')
    print(f"Saved plots {plots_titles}")


