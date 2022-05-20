from pathlib import Path
import warnings
import pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ks_2samp


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
    xbs, xrps = [], []
    for slide_id in tqdm(b_rp_slide_ids['b']):
        b_feature_path = next(path for path in features_paths if slide_id == path.with_suffix('').name)
        xb = pd.read_json(b_feature_path, orient='split')
        xbs.append(xb)
    for slide_id in tqdm(b_rp_slide_ids['rp']):
        rp_feature_path = next(path for path in features_paths if slide_id in path.name)
        xrp = pd.read_json(rp_feature_path, orient='split')
        xrps.append(xrp)
    x_all_bs, x_all_rps = pd.concat(xbs), pd.concat(xrps)
    test_values = []
    for feature_name in tqdm(x_all_bs.columns):
        s, p = ks_2samp(x_all_bs[feature_name], x_all_rps[feature_name])
        test_values.append({'feature_name': feature_name, 's': s, 'p': p})
    test_values = pd.DataFrame(test_values)
    test_values = test_values.set_index('feature_name')
    test_values.to_csv(features_dir/'ks.csv')
