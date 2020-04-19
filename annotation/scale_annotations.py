from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('annotations_path', type=Path)
    parser.add_argument('scale_factor', type=float)
    parser.add_argument('--save_original', action='store_true')
    args = parser.parse_args()
    for annotation_path in tqdm(list(args.annotations_path.iterdir())):
        if annotation_path.suffix != '.json':
            continue
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        if args.save_original:
            annotation.add_data('original', annotation._obj)
        annotation = annotation.scale(args.scale_factor)
        annotation.dump_to_json(args.annotations_path)
    print(f"Annotations were scaled and overwritten at {args.annotations_path}")