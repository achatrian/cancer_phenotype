import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


r"""Step preceding foci area extraction for ihc classification:
Merges slides' additional data for ihc and control cases
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ihc_file', type=Path)
    parser.add_argument('ihc_annotation_file', type=Path)
    parser.add_argument('--control_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/control/control_additional_data.csv')
    parser.add_argument('--control_annotation_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/control/control_annotation_data.csv')
    parser.add_argument('--old_batch_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/old_batch/Exported_additional_data_old_batch.csv')
    parser.add_argument('--old_annotation_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/old_batch/Exported_annotations_old_batch.csv')
    parser.add_argument('--target_dir', type=Path)
    args = parser.parse_args()

    with open(args.ihc_file, 'r') as ihc_file:
        ihc = pd.read_csv(ihc_file)
    ihc = ihc.drop('Benign/Malegnant', axis=1)  # TYPO 'malegnant' instead of 'malignant' in Nasullah's file
    with open(args.control_file, 'r') as control_file:
        control = pd.read_csv(control_file)
    control = control.drop('Diagnosis', axis=1).rename({'Benign/Malignant': 'Diagnosis'})
    with open(args.old_batch_file, 'r') as old_batch_file:
        old_batch = pd.read_csv(old_batch_file)
    old_batch = old_batch.drop('Benign/Malegnant', axis=1).rename({'Benign/Malignant': 'Diagnosis'})
    all_cases_data = pd.concat((ihc, control))
    all_cases_data.to_csv(args.target_dir/f'additional_data_{str(datetime.now())[:10]}.csv')
    print(f"Saved additional data for {len(all_cases_data)} foci in {args.target_dir}")

    with open(args.ihc_annotation_file, 'r') as ihc_annotation_file:
        ihc_annotation = pd.read_csv(ihc_annotation_file)
    with open(args.control_annotation_file, 'r') as control_annotation_file:
        control_annotation = pd.read_csv(control_annotation_file)
    with open(args.old_annotation_file, 'r') as old_annotation_file:
        old_annotation = pd.read_csv(old_annotation_file)
    all_annotations = pd.concat((ihc_annotation, control_annotation, old_annotation))
    all_annotations.to_csv(args.target_dir/f'annotations_{str(datetime.now())[:10]}.csv', index=False)
    print(f"Saved annotation data for {len(all_annotations)} cases in {args.target_dir}")
