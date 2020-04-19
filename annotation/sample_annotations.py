from pathlib import Path
from argparse import ArgumentParser
from random import sample
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder

r"""
Script to sample sample_size elements from a layer of annotations in a dataset 
"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--sample_size', type=int, default=30)
    parser.add_argument('--scale_factor', type=float, default=1.0)
    parser.add_argument('--layer', type=str, default='epithelium')
    parser.add_argument('--path_version', type=str, default='new', choices=['new', 'old'])
    args = parser.parse_args()
    annotation_dir = args.data_dir/'data'/'annotations'/args.experiment_name
    save_dir = args.data_dir/'data'/'annotations'/f'{args.experiment_name}_sampled_{args.sample_size}'
    save_dir.mkdir(exist_ok=True, parents=True)
    annotation_paths = tuple(annotation_dir.iterdir())
    for annotation_path in tqdm(annotation_paths):
        if not annotation_path.suffix == '.json':
            continue
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        try:
            sampling_layer = next(layer for layer in annotation._obj['layers'] if layer['name'] == args.layer)
        except StopIteration:
            raise
        sampled_items = sample(sampling_layer['items'], min(args.sample_size, len(sampling_layer['items'])))
        annotation.clear()
        sampling_layer['items'] = sampled_items
        # assert annotation.layers[sampling_layer['name']] is sampling_layer
        assert len(sampling_layer['items']) <= args.sample_size
        layer_idx = annotation.get_layer_idx(sampling_layer['name'])
        annotation._obj['layers'][layer_idx] = sampling_layer
        if args.scale_factor != 1.0:
            annotation.scale(args.scale_factor)
        if args.path_version == 'old':
            annotation.new_to_old()
        annotation.dump_to_json(save_dir)

