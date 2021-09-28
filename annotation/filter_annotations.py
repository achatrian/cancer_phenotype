import argparse
from pathlib import Path
import json
from random import shuffle
import numpy as np
import cv2
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation_dir', type=Path)
    parser.add_argument('area_annotation_dir', type=Path)
    parser.add_argument('--save_dir', type=Path)
    parser.add_argument('--shuffle_annotations', action='store_true')
    opt = parser.parse_args()
    annotations_paths = list(path for path in opt.annotation_dir.iterdir() if path.suffix == '.json')
    shuffle(annotations_paths)
    area_annotation_paths = list(path for path in opt.area_annotation_dir.iterdir() if path.suffix == '.json')
    opt.save_dir.mkdir(exist_ok=True)
    print(f"Filtering annotations in {str(opt.annotation_dir)} and saving to {str(opt.save_dir)}")
    for annotation_path in tqdm(annotations_paths):
        slide_id = annotation_path.name[:-5]
        if (opt.save_dir/f'{slide_id}.json').exists():
            continue
        try:  # load area annotation to filter with
            with open(opt.area_annotation_dir / (slide_id + '.json'), 'r') as annotation_file:
                annotation_obj = json.load(annotation_file)
                annotation_obj['slide_id'] = slide_id
                annotation_obj['project_name'] = 'tumour_area'
                annotation_obj['layer_names'] = ['Tumour area']
                contours, layer_name = AnnotationBuilder.from_object(annotation_obj). \
                    get_layer_points('Tumour area', contour_format=True)
        except ValueError as err:
            print([layer['name'] for layer in AnnotationBuilder.from_object(annotation_obj).layers.values()])
            print(err)
            continue
        except FileNotFoundError as err:
            print(f"No annotation file for '{slide_id}'")
            print(err)
            continue
        # biggest contour is used to select the area to process
        area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3),
                           key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(area_contour)

        def in_rect_check(contour, area_rect):
            contour_bounding_rect = cv2.boundingRect(contour)
            positions, _, __ = AnnotationBuilder.check_relative_rect_positions(contour_bounding_rect, area_rect)
            return positions in {'contained', 'overlap'}
        try:
            annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        except json.JSONDecodeError as err:
            print(f"Cannot decode annotation file for {slide_id}")
            print(err)
            continue
        for layer_name in annotation.layers:
            annotation.filter(layer_name, [in_rect_check], area_rect=(x, y, w, h))
        annotation.dump_to_json(opt.save_dir)
    print("Done!")
