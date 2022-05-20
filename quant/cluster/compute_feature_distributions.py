from pathlib import Path
from argparse import ArgumentParser
import json
from collections import namedtuple
import warnings
import copy
import numpy as np
from scipy.stats import iqr
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm


r"Compute distribution of extracted features"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--features_dir', type=Path, default=Path('/well/rittscher/projects/TCGA_prostate/TCGA/data/features/combined_mpp1.0_normal_nuclei_circles_good_nuc'))
    parser.add_argument('--num_bins', type=int, default=100)
    args = parser.parse_args()
    features_paths = list(args.features_dir.glob('*.json'))
    slides_ids = []
    distribution = namedtuple('distribution', ['id_', 'min', 'max', 'mean', 'std', 'iqr', 'q5', 'q95',
                                               'n', 'num_bins', 'distribution', 'columns'])
    distributions = []
    # first loop to measure dataset properties and find best parameters to estimate probability mass
    for feature_path in tqdm(features_paths, desc='first pass'):
        slide_id = feature_path.with_suffix('').name
        try:
            with open(feature_path, 'r') as features_file:
                x = pd.read_json(features_file, orient='split')
        except ValueError:
            tqdm.write(f"Could not read features for slide {slide_id}")
            continue
        distributions.append(distribution(
            id_=slide_id,
            min=x.min(axis=0),
            max=x.max(axis=0),
            mean=x.mean(axis=0),
            std=x.std(axis=0),
            iqr=x.apply(iqr, axis=0),
            q5=x.quantile(0.05, axis=0),
            q95=x.quantile(0.95, axis=0),
            n=x.shape[0],
            num_bins=args.num_bins,
            distribution=None,
            columns=x.columns.tolist()
        ))
    dataset_distribution = distribution(
        id_='dataset',
        min=pd.concat([d.min for d in distributions], axis=1).T.min(axis=0),
        max=pd.concat([d.max for d in distributions], axis=1).T.max(axis=0),
        mean=pd.concat([d.mean for d in distributions], axis=1).T.median(axis=0),
        std=pd.concat([d.mean for d in distributions], axis=1).T.median(axis=0),
        iqr=pd.concat([d.iqr for d in distributions], axis=1).T.median(axis=0),
        q5=pd.concat([d.q5 for d in distributions], axis=1).T.min(axis=0),
        q95=pd.concat([d.q95 for d in distributions], axis=1).T.max(axis=0),
        n=sum(d.n for d in distributions),
        num_bins=args.num_bins,
        distribution=None,
        columns=distributions[0].columns
    )
    # second loop to compute histograms

    def dataset_histogram(column: pd.Series, num_bins):
        name = column.name
        # n = dataset_distribution.n/len(distributions)  # average number of data-points per slide
        # h = 2*dataset_distribution.iqr[name]*n**(1/3)  # bin width from the Freedman-Diaconis rule
        # if h < 0.0001:
        #     h = 1
        return np.histogram(column,
                            bins=num_bins,
                            range=(dataset_distribution.q5[name], dataset_distribution.q95[name]))[0]
    for feature_path in tqdm(features_paths, desc='second pass'):
        slide_id = feature_path.with_suffix('').name
        try:
            with open(feature_path, 'r') as features_file:
                x = pd.read_json(features_file, orient='split')
        except ValueError:
            tqdm.write(f"Could not read features for slide {slide_id}")
            continue
        index = next(i for i, d in enumerate(distributions) if d.id_ == slide_id)
        distributions[index] = distributions[index]._replace(
            distribution=x.apply(dataset_histogram, axis=0, args=(args.num_bins,)).to_json(orient='records'),

            min=distributions[index].min.to_json(orient='records'),
            max=distributions[index].max.to_json(orient='records'),
            mean=distributions[index].mean.to_json(orient='records'),
            std=distributions[index].std.to_json(orient='records'),
            iqr=distributions[index].iqr.to_json(orient='records'),
            q5=distributions[index].q5.to_json(orient='records'),
            q95=distributions[index].q95.to_json(orient='records')
        )
    distributions.append(dataset_distribution._replace(
        min=dataset_distribution.min.to_json(orient='records'),
        max=dataset_distribution.max.to_json(orient='records'),
        mean=dataset_distribution.mean.to_json(orient='records'),
        std=dataset_distribution.std.to_json(orient='records'),
        iqr=dataset_distribution.iqr.to_json(orient='records'),
        q5=dataset_distribution.q5.to_json(orient='records'),
        q95=dataset_distribution.q95.to_json(orient='records')
    ))
    with open(args.features_dir.parent/f'{args.features_dir.name}_distribution_{args.num_bins}bins.json', 'w') as distribution_file:
        json.dump(distributions, distribution_file)
    print(f"{len(distributions)} slide-level feature distributions were estimated.")
    print("Done!")







