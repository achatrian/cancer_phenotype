import sys
sys.path.extend(['/well/rittscher/users/achatrian/cancer_phenotype/base',
                 '/well/rittscher/users/achatrian/cancer_phenotype'])
from pathlib import Path
import json
from pandas import DataFrame, concat
from base.options.train_options import TrainOptions

"""
Script to merge all the resolution.json files in the tile slide folders into one file that can be read and cached by dataset
"""


if __name__ == '__main__':
    sys.argv.append('--data_dir=/well/rittscher/projects/TCGA_prostate/TCGA')
    opt = TrainOptions().parse()
    root_path = Path(opt.data_dir)
    resolution_paths = sorted(
        root_path.glob('./data/tiles/*/resolution.json'),
        key=lambda path: str(path.parent.name)
    )  # sorted by slide id
    resolutions = dict()
    statistics = dict(target_mpp=[], target_qc_mpp=[], read_level=[], qc_read_level=[], read_mpp=[], qc_mpp=[])
    for resolution_path in resolution_paths:
        with open(resolution_path, 'r') as resolution_file:
            resolution_data = json.load(resolution_file)
        resolutions[resolution_path.parent.name] = resolution_data
        for key, value in resolution_data.items():
            if key not in ('tissue_locations',):
                statistics[key].append(value)
    with open(root_path/'data'/'CVsplits'/'tcga_resolution.json', 'w') as resolution_file:
        json.dump(resolutions, resolution_file)
    with open(root_path/'data'/'CVsplits'/'tcga_resolution_stats.csv', 'w') as resolution_stats_file:
        df = DataFrame.from_dict(statistics)
        concat((
            df.aggregate(['mean', 'std']),
            df
        )).to_csv(resolution_stats_file)
    print("Done !")




