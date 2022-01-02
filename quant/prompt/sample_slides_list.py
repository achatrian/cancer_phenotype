from pathlib import Path
import pandas as pd
import json
from collections import Counter
from matplotlib import pyplot as plt
from numpy import array


if __name__ == '__main__':
    TARGET_N = 1500
    master_list_path = Path('/mnt/rescomp/projects/ProMPT/ProMPT_master_list.xlsx')
    master_list = pd.read_excel(master_list_path)
    slides_list = pd.read_excel(master_list_path, sheet_name='slides')
    with open('/mnt/rescomp/projects/ProMPT_data/last_biopsy_rp_pairs.json', 'r') as pairs_file:
        last_biopsy_rp_pairs = json.load(pairs_file)
    pairs = list(pair_obj['cases'] for pair_obj in last_biopsy_rp_pairs)
    pairs_dict = dict(pairs)
    pairs_dict.update(tuple(reversed(pair) for pair in pairs))
    common_cases = set(master_list['SpecimenIdentifier']).intersection(set(slides_list['SpecimenIdentifier']))
    master_list = master_list[master_list['SpecimenIdentifier'].isin(common_cases)]
    # master_list = master_list[master_list['SpecimenIdentifier'].isin(set(id_ for pair in pairs for id_ in pair))]  # this removes wrong sldies for some reason
    patients_specimens_path = Path('/mnt/rescomp/users/achatrian/ProMPT_patient_specimens.xlsx')
    patients_specimens_data = pd.read_excel(patients_specimens_path)
    biopsy_vs_rp_data = patients_specimens_data[(patients_specimens_data['#_biopsies'] > 0)
                                                & (patients_specimens_data['#_RPs'] > 0)]
    biopsy_vs_rp_cases = master_list[master_list['PromptID'].isin(biopsy_vs_rp_data['prompt_id'])].copy()
    print(f"{len(biopsy_vs_rp_cases)} cases for patients having both biopsy and RP data")

    def assign_gleason_label(specimen_data):
        if specimen_data['TotalGleason'] in {6, 8, 9, 10}:
            return str(int(specimen_data['TotalGleason']))
        elif specimen_data['PrimaryGleason'] == 3 and specimen_data['SecondaryGleason'] == 4:
            return '3+4'
        elif specimen_data['PrimaryGleason'] == 4 and specimen_data['SecondaryGleason'] == 3:
            return '4+3'
        else:
            return 'NA'
    biopsy_vs_rp_cases['gleason_label'] = biopsy_vs_rp_cases.apply(assign_gleason_label, axis=1)
    biopsy_vs_rp_cases = biopsy_vs_rp_cases[biopsy_vs_rp_cases['gleason_label'] != 'NA']
    biopsy_vs_rp_slides = slides_list[slides_list['SpecimenIdentifier'].isin(biopsy_vs_rp_cases['SpecimenIdentifier'])].copy()
    print(f"{len(biopsy_vs_rp_slides)} slides for patients having both biopsy and RP data")
    if len(biopsy_vs_rp_slides) < TARGET_N:
        raise ValueError("Sampling N bigger than data length")
    gleason_fractions = biopsy_vs_rp_cases.groupby('gleason_label').size().to_frame().squeeze()/len(biopsy_vs_rp_cases)
    biopsy_vs_rp_cases.set_index('SpecimenIdentifier', inplace=True)
    biopsy_vs_rp_cases['#_slides'] = biopsy_vs_rp_slides.groupby('SpecimenIdentifier').size().to_frame().squeeze()
    counts = Counter({'6': 0, '3+4': 0, '4+3': 0, '8': 0, '9': 0})
    num_slides = 0
    sampled = []
    biopsy_vs_rp_cases_copy = biopsy_vs_rp_cases.copy(deep=True)
    errors_history = {'6': [], '3+4': [], '4+3': [], '8': [], '9': []}
    while num_slides < TARGET_N:
        # weights = [gleason_fractions.loc[row['gleason_label']] -
        #            counts[row['gleason_label']]/len(biopsy_vs_rp_cases)
        #            for i, row in biopsy_vs_rp_cases_copy.iterrows()]
        errors = {gleason: max(1 - counts[gleason]/(sum(counts.values()) + 0.0001)/gleason_fractions.loc[gleason], 0)
                  for gleason in counts.keys()}
        for gleason, error in errors.items():
            errors_history[gleason].append(error)
        max_error_gleason_level = max(errors.items(), key=lambda e: e[1])[0]
        case = biopsy_vs_rp_cases_copy[biopsy_vs_rp_cases_copy['gleason_label'] == max_error_gleason_level].sample(1)
        # case = biopsy_vs_rp_cases_copy.sample(1, weights=weights)
        try:
            paired_case = biopsy_vs_rp_cases_copy.loc[[pairs_dict[case.index.item()]]]  # give list to loc to return DataFrame
        except KeyError as err:
            print(err)
            continue
        case['paired_case'] = paired_case.index.item()
        paired_case['paired_case'] = case.index.item()
        if case['SpecimenType'].item() == 'Biopsy':
            sampled.extend([case, paired_case])
        else:
            sampled.extend([paired_case, case])
        len_1 = len(biopsy_vs_rp_cases_copy)
        biopsy_vs_rp_cases_copy = biopsy_vs_rp_cases_copy.drop([case.index.item(), paired_case.index.item()])
        assert len(biopsy_vs_rp_cases_copy) != len_1
        num_slides += (case['#_slides'].item() + paired_case['#_slides'].item())
        counts.update([case['gleason_label'].item(), paired_case['gleason_label'].item()])
        plt.plot(array([e for e in errors_history.values()]).T)
        plt.ylim((0, 1))
        plt.title('errors')
        plt.show()
    sampled_cases = pd.concat(sampled)
    sampled_cases.to_excel('/mnt/rescomp/users/achatrian/ProMPT_biopsy_vs_rp_cases_sample.xlsx')
    sampled_gleason_fractions = sampled_cases.groupby('gleason_label').size().to_frame().squeeze()/len(sampled_cases)
    print("Gleason fractions full data:")
    print(gleason_fractions)
    print("Gleason fractions sampled data:")
    print(sampled_gleason_fractions)
    print(f"{len(sampled_cases)} cases were selected")
    print("Done!")



