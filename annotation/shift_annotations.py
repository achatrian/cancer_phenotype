from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('annotations_path', type=Path)
    parser.add_argument('shift_x', type=int)
    parser.add_argument('shift_y', type=int)
    parser.add_argument('--save_original', action='store_true')
    parser.add_argument('--save_dir', type=Path, default=None)
    args = parser.parse_args()
    for annotation_path in tqdm(list(args.annotations_path.iterdir())):
        if annotation_path.suffix != '.json':
            continue
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        if args.save_original:
            annotation.add_data('original', annotation._obj)
        annotation = annotation.shift((args.shift_x, args.shift_y))
        if args.save_dir is not None:
            annotation.dump_to_json(args.save_dir/args.annotations_path.name)
        else:
            annotation.dump_to_json(args.annotations_path)
    print(f"Annotations were scaled and overwritten at {args.annotations_path}")
