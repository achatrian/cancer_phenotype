import argparse
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=Path)
    parser.add_argument('-t-', '--target_dir', type=Path, default=None)
    parser.add_argument('--merge_annotators', action='store_true')
    args = parser.parse_args()
    with args.csv_path.open('r') as csv_file:
        annotations_sheet = pd.read_csv(csv_file)  # need quotechar that does not appear in json
    for index, row in tqdm(annotations_sheet.iterrows(), total=len(annotations_sheet)):  # minus header row
        obj = json.loads(row['Annotation'])
        annotator_name = row['Annotator']
        image_name = row['Image']
        obj = obj['annotation']
        if args.merge_annotators:
            save_dir = args.target_dir if args.target_dir is not None else args.csv_path.parent
        else:
            if args.target_dir is not None:
                save_dir = args.target_dir / annotator_name
            else:
                save_dir = args.csv_path.parent / annotator_name
        save_dir.mkdir(exist_ok=True, parents=True)
        with open(save_dir/f'{image_name}.json', 'w') as annotation_file:
            json.dump(obj, annotation_file)
    print("Done !")
