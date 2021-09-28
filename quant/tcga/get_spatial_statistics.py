from pathlib import Path
from argparse import ArgumentParser
from numpy import histogram, linspace
from pandas import read_json, DataFrame
from tqdm import tqdm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser.add_argument('--experiment_name', type=str, required=True,
                                 help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    parser.add_argument('--max_distance', type=int, default=100000)
    parser.add_argument('--num_bins', type=int, default=100)
    args = parser.parse_args()

    distances_dir = args.data_dir/'data'/'features'/args.experiment_name/'relational'
    bins = linspace(0, args.max_distance, args.num_bins+1)

    # get max and minimum distance across
    histograms = {}
    distances_paths = list(distances_dir.iterdir())
    errors = []
    for distances_path in tqdm(distances_paths):
        slide_id = distances_path.name.split('_')[1][:-5]
        try:
            distances = read_json(distances_path)
            hist = histogram(distances, bins, range=(0, args.max_distance))[0].tolist()
            histograms[slide_id] = hist
        except ValueError as err:
            print(f"Error for {slide_id}: '{err}'")
            histograms[slide_id] = [0]*args.num_bins
    histograms = DataFrame.from_dict(histograms, 'index')
    histograms.index.rename('slide_id', inplace=True)
    histograms.to_json(distances_dir/'distance_distributions.json')
    print("Done!")
    print(f"Num reading errors = {len(errors)}")
