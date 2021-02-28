from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from copy import copy


r"""Create validation data that's compatible with training/test annotations for the validation datatset"""


# cases_to_discard = (
#     '10392_19_3_L3',  # presence of many small regions of broken tissue
#     '12990_19_1_L3',  # very small slide / piece of tissue
#     '19494_19_3_L3',  # small piece of tissue
#     '17997_19_1_L3',  # very thin slide: mask detection failed
#     '11846_19_2_L3',  # very thin slide: mask detection failed
#     '14138_19_3_L3',  # very small slide
#     '17948_19_2_L3_1',  # mask detection failed
#     '8030_19_1_L3',  # presence of many small regions of broken tissue
#     '17948_19_1_L3',  # very thin slide: mask detection failed
#     '17954_19_1_L3',  # very thin slide: mask detection failed
#     '14138_19_3_L3',  # very small slide
# )
# these cases were misclassified because they were missing labels!


if __name__ == '__main__':
    validation_annotations_clare = pd.read_csv('/Users/andreachatrian/Mounts/rescomp/gpfs2/well/rittscher/projects/IHC_Request/data/documents/final_validation_annotations/clares_annotations.csv')
    validation_annotations_richard = pd.read_csv('/Users/andreachatrian/Mounts/rescomp/gpfs2/well/rittscher/projects/IHC_Request/data/documents/final_validation_annotations/richards_annotations.csv')
    validation_annotations_lisa = pd.read_csv('/Users/andreachatrian/Mounts/rescomp/gpfs2/well/rittscher/projects/IHC_Request/data/documents/final_validation_annotations/lisas_annotations.csv')

    validation_data_clare = validation_annotations_clare[['Image', 'IHC request', 'Annotator']]
    validation_data_clare = validation_data_clare.rename(columns={'IHC request': 'Case type'})
    validation_data_clare = validation_data_clare.replace('Yes', 'Real')
    validation_data_clare = validation_data_clare.replace('No', 'Control')
    validation_data_clare = validation_data_clare.dropna()


    validation_data_richard = validation_annotations_richard[['Image', 'IHC request', 'Annotator']]
    validation_data_richard = validation_data_richard.rename(columns={'IHC request': 'Case type'})
    validation_data_richard = validation_data_richard.replace('Yes', 'Real')
    validation_data_richard = validation_data_richard.replace('No', 'Control')
    validation_data_richard = validation_data_richard.dropna()


    validation_data_lisa = validation_annotations_lisa[['Image', 'IHC request', 'Annotator']]
    validation_data_lisa = validation_data_lisa.rename(columns={'IHC request': 'Case type'})
    validation_data_lisa = validation_data_lisa.replace('Yes', 'Real')
    validation_data_lisa = validation_data_lisa.replace('No', 'Control')
    validation_data_lisa = validation_data_lisa.dropna()


    slide_ids_annotated_all = set.intersection(set(validation_data_clare['Image']), set(validation_data_richard['Image']),
                                        set(validation_data_lisa['Image']))
    validation_data_clare = validation_data_clare[validation_data_clare['Image'].isin(slide_ids_annotated_all)]
    validation_data_richard = validation_data_richard[validation_data_richard['Image'].isin(slide_ids_annotated_all)]
    validation_data_lisa = validation_data_lisa[validation_data_lisa['Image'].isin(slide_ids_annotated_all)]
    validation_data_clare.to_csv('/Users/andreachatrian/Mounts/rescomp/gpfs2/well/rittscher/projects/IHC_Request/data/documents/validation_data_clare.csv')
    validation_data_richard.to_csv('/Users/andreachatrian/Mounts/rescomp/gpfs2/well/rittscher/projects/IHC_Request/data/documents/validation_data_richard.csv')
    validation_data_lisa.to_csv('/Users/andreachatrian/Mounts/rescomp/gpfs2/well/rittscher/projects/IHC_Request/data/documents/validation_data_lisa.csv')


    validation_data_consensus = validation_data_clare.copy()
    for index, row in validation_data_clare.iterrows():
        slide_id = row['Image']
        p1 = validation_data_clare[validation_data_clare['Image'] == slide_id]['Case type'] == 'Real'
        p2 = validation_data_richard[validation_data_richard['Image'] == slide_id]['Case type'] == 'Real'
        p3 = validation_data_lisa[validation_data_lisa['Image'] == slide_id]['Case type'] == 'Real'
        try:
            if p1.iloc[0] + p2.iloc[0] + p3.iloc[0] >= 2:
                validation_data_consensus[validation_data_consensus['Image'] == slide_id]['Case type'] = 'Real'
        except IndexError:
            continue
        validation_data_consensus[validation_data_consensus == 'Image']['Annotator'] = 'consensus'
    validation_data_lisa.to_csv('/Users/andreachatrian/Mounts/rescomp/gpfs2/well/rittscher/projects/IHC_Request/data/documents/validation_data_consensus.csv')



