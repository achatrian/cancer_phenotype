from pathlib import Path
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder

r"""
Script to sample sample_size elements from a layer of annotations in a dataset 
"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--stroke_width', type=int, default=60)
    parser.add_argument('--layer', type=str, default='all')
    args = parser.parse_args()
    annotation_dir = args.data_dir/'data'/'annotations'/args.experiment_name
    save_dir = args.data_dir/'data'/'annotations'/f'{args.experiment_name}_stroke_{args.stroke_width}'
    save_dir.mkdir(exist_ok=True, parents=True)
    annotation_paths = tuple(annotation_dir.iterdir())
    for annotation_path in tqdm(annotation_paths):
        if not annotation_path.suffix == '.json':
            continue
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        layers = [name for name in annotation.layers.keys() if name == args.layer or args.layer == 'all']
        for name in layers:
            annotation.set_items_attribute(name, 'strokeWidth', args.stroke_width)
            annotation.set_items_attribute(name, 'strokeScaling', True)
        annotation.dump_to_json(save_dir)
    print("Done!")




