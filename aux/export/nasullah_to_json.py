import argparse
from pathlib import Path
import csv
import json
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=Path)
    args = parser.parse_args()
    with args.csv_path.open('r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='@')  # need quotechar that does not appear in json
        next(reader)  # first row is description of how data is stored --> skip
        for row in tqdm(reader):
            image_name = row[0].replace('"', '')  # get rid of extra quotes
            annotation_content = ','.join(row[1:-1])[1:-1]  # there are extra "" at beginning and end of annotation
            if annotation_content[-1] != '}':
                # some annotations also have a completion status flag
                annotation_content = ','.join(row[1:-2])[1:-1]
                annotator_name = row[-2].replace('"', '')  # in which case the annotator's name is the penultimate entry
                status = row[-1].replace('"', '')
            else:
                annotator_name = row[-1].replace('"', '')
                status = None
            try:
                obj = json.loads(annotation_content)
            except json.decoder.JSONDecodeError as decode_err0:
                try:
                    if '""' in annotation_content:  # FIXME this breaks for empty strings (which remain open)
                        annotation_content = annotation_content.replace('""', '"')  # some annotation file are read with double double quotes?
                    obj = json.loads(annotation_content)
                except json.decoder.JSONDecodeError as decode_err1:
                    raise NotImplementedError("Empty strings in json (cannot deal with this yet)")
            obj = obj['annotation']
            annotator_dir = args.csv_path.parent/annotator_name
            annotator_dir.mkdir(exist_ok=True)
            with open(annotator_dir/f'{image_name}.json', 'w') as annotation_file:
                json.dump(obj, annotation_file)
    print("Done !")
