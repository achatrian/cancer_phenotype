from pathlib import Path
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kstest, wasserstein_distance
from scipy.spatial.distance import jensenshannon


if __name__ == '__main__':
    features_dir = Path('/well/rittscher/projects/ProMPT/data/features/combined_mpp1.0_normal_nuclei_circles_good_nuc')
    features_paths = list(features_dir.glob('*.json'))
    master_list = pd.read_excel('/well/rittscher/projects/ProMPT/cases/ProMPT_master_list.xlsx', sheet_name='cases')
    master_list = master_list.set_index('SpecimenIdentifier')
    b_rp_labels = np.array(['b' if master_list.loc[int(path.name.split('_')[0])]['SpecimenType'] == 'Biopsy' else 'rp'
                            for path in features_paths])
    b_rp_slide_ids = {'b': [], 'rp': []}
    for feature_path in features_paths:
        case_id = int(feature_path.name.split('_')[0])
        if master_list.loc[case_id, 'SpecimenType'] == 'Biopsy':
            b_rp_slide_ids['b'].append(feature_path.with_suffix('').name)
        if master_list.loc[case_id, 'SpecimenType'] == 'RP':
            b_rp_slide_ids['rp'].append(feature_path.with_suffix('').name)
    slide_ids = b_rp_slide_ids['b'] + b_rp_slide_ids['rp']
    ks_statistics, ks_pvalues, wasserstein_distances, js_divergences = {}, {}, {}, {}
    for slide_id0 in tqdm(slide_ids, desc='slide ids outer loop'):
        b_feature_path = next(path for path in features_paths if slide_id0 == path.with_suffix('').name)
        xb = pd.read_json(b_feature_path, orient='split')
        for slide_id1 in tqdm(slide_ids, desc='slide ids inner loop'):
            # kolmogorov - smirnov and wasserstein distance
            rp_feature_path = next(path for path in features_paths if slide_id1 in path.name)
            xrp = pd.read_json(rp_feature_path, orient='split')
            ks_p = [kstest(xb[feature_name].to_numpy().squeeze(), xrp[feature_name].to_numpy().squeeze()) for feature_name in xb]
            ks_statistics[f'{slide_id0},{slide_id1}'] = {feature_name: s_p[0] for feature_name, s_p in zip(xrp, ks_p)}
            ks_pvalues[f'{slide_id0},{slide_id1}'] = {feature_name: s_p[1] for feature_name, s_p in zip(xrp, ks_p)}
            wasserstein_distances[f'{slide_id0},{slide_id1}'] = {feature_name: wasserstein_distance(xb[feature_name], xrp[feature_name]) for feature_name in xb}
            # kl divergence (relative entropy)
            bin_edges = {feature_name: np.histogram_bin_edges(pd.concat((xb, xrp), axis=0)[feature_name]) for feature_name in xb}
            js_divergences[f'{slide_id0},{slide_id1}'] = {feature_name: jensenshannon(
                np.histogram(xb[feature_name], bins=bin_edges[feature_name])[0],
                np.histogram(xrp[feature_name], bins=bin_edges[feature_name])[0]
            ) for feature_name in xb}
    pd.DataFrame(ks_statistics).to_csv(features_dir.parent/'ks_statistics.csv')
    pd.DataFrame(ks_pvalues).to_csv(features_dir.parent/'ks_pvalues.csv')
    pd.DataFrame(wasserstein_distances).to_csv(features_dir.parent/'wasserstein_distances.csv')
    pd.DataFrame(js_divergences).to_csv(features_dir.parent/'js_divergences.csv')
    print("Done!")