from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


r"""Find coverage of biopsy slides over radical prostatectomy slides"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--features_dir', type=Path, default='/mnt/rescomp/projects/ProMPT/data/features/combined_mpp1.0_normal_nuclei_circles_good_nuc')
    parser.add_argument('--percentile', type=int, default=3)
    parser.add_argument('--save_dir', type=Path, default='/mnt/rescomp/projects/ProMPT/data/results')
    # parser.add_argument('--r_min', type=int, default=100)
    args = parser.parse_args()
    features_paths = list(args.features_dir.glob('*.json'))
    features, slides_ids_ = [], []
    for feature_path in tqdm(features_paths):
        f = pd.read_json(feature_path, orient='split', convert_axes=False)
        features.append(f)
        slides_ids_.extend([feature_path.with_suffix('').name]*len(f))
    features = pd.concat(features)
    features.index = pd.MultiIndex.from_arrays([slides_ids_, features.index])
    slides_ids = list(set(slides_ids_))
    master_list = pd.read_excel('/mnt/rescomp/projects/ProMPT/cases/ProMPT_master_list.xlsx', sheet_name='cases')
    master_list = master_list.set_index('SpecimenIdentifier')
    biopsies, rps = [], []
    for slide_id in slides_ids:
        case_id = int(slide_id.split('_')[0])
        if master_list.loc[case_id, 'SpecimenType'] == 'Biopsy':
            biopsies.append(slide_id)
        else:
            rps.append(slide_id)
    with open('/mnt/rescomp/projects/ProMPT/cases/pairs.json', 'r') as pairs_file:
        pairs = json.load(pairs_file)
    pairs = [p['cases'] for p in pairs]
    assert all(master_list.loc[p[0], 'SpecimenType'] == 'Biopsy' and master_list.loc[p[1], 'SpecimenType'] == 'RP' for p in pairs)
    scaler = MinMaxScaler().fit(features)
    scaled_features = pd.DataFrame(scaler.transform(features), index=features.index, columns=features.columns)
    # rp_features = scaled_features.loc[(rps,), :]
    # b_features = scaled_features.loc[(biopsies,), :]
    D = pairwise_distances(scaled_features.sample(2000), metric='cityblock')  # use cityblock distance to evaluate features separately
    radius = np.percentile(D, args.percentile)

    def coverage(points, bt: NearestNeighbors):
        indices = bt.radius_neighbors(points, radius, return_distance=False)  # returns array of arrays of unequal length
        indices = set(np.concatenate(indices))
        return len(indices)/len(bt._fit_X)
    data = {'pair_rp': [], 'pair_biopsy': []}
    # check coverage for biopsy over rp, and additionally for rp over rp as a sanity check
    biopsies_rps = biopsies + rps  # biopsies first, same order as in master list
    data.update({slide_id: [] for slide_id in biopsies_rps})
    for rp_id in tqdm(rps, desc='rp'):
        try:
            b_case = next(p[0] for p in pairs if p[1] == int(rp_id.split('_')[0]))
            b_id = next(b_id_ for b_id_ in biopsies if int(b_id_.split('_')[0]) == b_case)
        except StopIteration:
            continue
        # compute coverage over paired biopsy and RP
        f_rp = scaled_features.xs(rp_id)
        bt_rp = NearestNeighbors(n_neighbors=10, metric='cityblock').fit(f_rp)
        c = {slide_id: coverage(scaled_features.xs(slide_id), bt_rp) for slide_id in tqdm(biopsies_rps, desc='all')}
        tqdm.write(f'{rp_id}: {c[b_id]:.2f}, {np.mean(list(c.values())):.2f}, {np.median(list(c.values())):.2f}, {np.min(list(c.values())):.2f}, {np.max(list(c.values())):.2f}')  # TODO print this over the biopsies?
        data['pair_rp'].append(rp_id), data['pair_biopsy'].append(b_id)
        for slide_id in biopsies_rps:
            data[slide_id].append(c[slide_id])
    df = pd.DataFrame(data)
    df = df.set_index(['pair_rp', 'pair_biopsy'])
    df.columns = pd.MultiIndex.from_arrays([['biopsy']*len(biopsies) + ['rp']*len(rps), df.columns])
    df.to_excel(args.save_dir/f'rp_coverage_{args.percentile}.xlsx')
    print("Done!")


    # f_b_pca = pca.transform(scaler.transform(f_b))
    # f_rp_pca = pca.transform(scaler.transform(f_rp))
    # for
    #
    # x_b2 = pd.read_json(next(features_dir.glob(f'{b_id2}.json')), orient='split')
    #
    # x_b2_pca = pca.transform(scaler.transform(x_b2))


