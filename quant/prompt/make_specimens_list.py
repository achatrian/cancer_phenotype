from pathlib import Path
from collections import defaultdict
import pandas as pd
from math import isnan


if __name__ == '__main__':
    master_list_path = Path('/mnt/rescomp/projects/ProMPT/ProMPT_master_list.xlsx')
    master_list = pd.read_excel(master_list_path)
    patients_specimens = defaultdict(lambda: [])
    for index, specimen_data in master_list.iterrows():
        prompt_id, specimen_id, specimen_type = specimen_data['PromptID'], specimen_data['SpecimenIdentifier'], specimen_data['SpecimenType']
        gleason_primary, gleason_secondary, gleason_total = specimen_data['PrimaryGleason'], specimen_data['SecondaryGleason'], specimen_data['TotalGleason']
        gleason_primary = int(gleason_primary) if not isnan(gleason_primary) else ''
        gleason_secondary = int(gleason_secondary) if not isnan(gleason_secondary) else ''
        gleason_total = int(gleason_total) if not isnan(gleason_total) else ''
        scanned = 'scanned' if specimen_data['Scanned'] else 'unscanned'
        specimen_repr = f'{specimen_id}:{specimen_type}:{gleason_primary}:{gleason_secondary}:{gleason_total}:{scanned}'
        patients_specimens[prompt_id].append(specimen_repr)
    patients_specimens_data = [(prompt_id,
                                len(specimens),
                                sum('Biopsy' in specimen for specimen in specimens),
                                sum('RP' in specimen for specimen in specimens),
                                sum('TURP' in specimen for specimen in specimens),
                                '; '.join(specimens)) for prompt_id, specimens in patients_specimens.items()]
    patients_specimens_data = pd.DataFrame(patients_specimens_data,
                                           columns=['prompt_id', '#_specimens', '#_biopsies', '#_RPs', '#_TURPs', 'specimens'])
    patients_specimens_path = Path('/mnt/rescomp/users/achatrian/ProMPT_patient_specimens.xlsx')
    patients_specimens_data.to_excel(patients_specimens_path)
    print("Done!")

