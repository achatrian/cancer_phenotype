from pathlib import Path
import json
import copy
from matplotlib import pyplot as plt
import pandas as pd


# 30/05/2019 first look at features (but those extracted before fixing mask to contour) -- waiting for correct annotations to be extracted from dzi's
# - featues must be saved / loaded with orient='index'


if __name__ == '__main__':
    feature_dir = Path('/mnt/rescomp/projects/prostate-gland-phenotyping/WSI/data/features')
    feature_path = feature_dir/'17_A047-4519_1614P+-+2017-05-11+09.50.49.json'
    with open(feature_path, 'r') as features_file:
        data = json.load(features_file)
    slide_id = copy.deepcopy(data['slide_id'])
    tissue_data = copy.deepcopy(data['data'])
    dist = copy.deepcopy(data['dist'])
    del data['slide_id'], data['data'], data['dist']
    x = pd.DataFrame.from_dict(data, orient='index')
    print(x.columns)


