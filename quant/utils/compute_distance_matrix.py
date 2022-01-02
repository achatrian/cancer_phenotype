from pathlib import Path
from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
from pandas import DataFrame


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser.add_argument('experiment_name', type=str, help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    parser.add_argument('-ds', '--debug_slide', action='append', help="")
    args = parser.parse_args()

    feature_dir = args.data_dir/'data'/'features'/args.experiment_name
    contours_data_dir = feature_dir/'data'
    contours_data_paths = tuple(contours_data_dir.iterdir())
    if args.debug_slide is not None and len(args.debug_slide) > 0:
        contours_data_paths = [path for path in contours_data_paths if path.with_suffix('').name[5:] in args.debug_slide]
    assert contours_data_paths, "non-empty dataset"
    for data_path in tqdm(contours_data_paths):
        slide_id = '_'.join(data_path.name.split('_')[1:])[:-5]
        try:
            with data_path.open('r') as data_file:
                contours_data = json.load(data_file)
        except ValueError as err:
            print(f"Error for {slide_id}: '{err}'")
        centroids = np.array(list(set(tuple(datum['centroid']) for datum in contours_data)))  # remove duplicates
        tqdm.write(f"{slide_id}: {len(centroids)} contours")
        dist = DataFrame(distance.cdist(centroids, centroids, 'euclidean'),
                         index=tuple(f'{c[0]}_{c[1]}' for c in centroids),
                         columns=tuple(f'{c[0]}_{c[1]}' for c in centroids))
        with open(feature_dir/'relational'/('dist_' + slide_id + '.json'), 'w') as dist_file:
            dist.to_json(dist_file)
    print("Done!")



