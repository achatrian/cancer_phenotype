from pathlib import Path
import argparse
import json
import numpy as np
import cv2
from base.utils.annotation_builder import AnnotationBuilder


def read_annotations(slide_ids, data_dir):
    """
    Read annotations for one / many slides
    :return: dict: annotation id --> (dict: layer_name --> layer points)
    """
    assert type(slide_ids) in (tuple, list, set)
    slide_ids = set(slide_ids)
    annotation_dir = Path(data_dir)/'data'/'annotations'
    if slide_ids:
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()
                            if any(slide_id in str(annotation_path.name) for slide_id in slide_ids)]
    contour_struct = dict()
    for annotation_path in annotation_paths:
        with open(annotation_path, 'r') as annotation_file:
            annotation_obj = json.load(annotation_file)
            annotation = AnnotationBuilder.from_object(annotation_obj)
        annotation_id = annotation_path.name.replace('.json', '')
        contour_struct[annotation_id] = dict()
        for layer_name in annotation.layers:
            contour_struct[annotation_id][layer_name], _ = annotation.get_layer_points(layer_name, contour_format=True)
    return contour_struct


def find_overlap(slide_contours: dict):
    """Returns structure detailing which contours overlap, so that overlapping features can be computed"""
    contours, labels = [], []
    for layer_name, layer_contours in slide_contours.items():
        contours.extend(layer_contours)
        labels.extend([layer_name] * len(layer_contours))
    contour_bbs = list(cv2.boundingRect(contour) for i, contour in enumerate(contours))
    overlap_struct = []
    for parent_bb in contour_bbs:
        # Find out each contour's children (overlapping)
        overlap_vector = []
        for child_bb in contour_bbs:
            position, origin_bb, bb_areas = AnnotationBuilder.check_relative_rect_positions(parent_bb, child_bb, eps=0)
            if parent_bb == child_bb:  # don't do anything for same box
                overlap_vector.append(False)
            elif position == 'contained' and origin_bb == 0:
                overlap_vector.append(True)
            elif position == 'overlap':
                overlap_vector.append(True)
            else:
                overlap_vector.append(False)
        overlap_struct.append(overlap_vector)
    return overlap_struct


def contour_to_mask(contour: np.ndarray, value=255, shape=()):
    """
    Convert a contour to the corresponding max - mask is
    :param contour:
    :param value:
    :param shape
    :return:
    """
    assert type(contour) is np.ndarray
    contour = contour.squeeze()
    if not shape:
        shape = (contour[:, 0].max() - contour[:, 0].min(), contour[:, 1].max() - contour[:, 1].min())  # dimension of contour
    mask = np.ones(shape)
    cv2.drawContours(mask, [contour], -1, value, thickness=-1)  # thickness=-1 fills the entire area inside
    return mask



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slide_ids', type=str, nargs='+', help="Slide ids to process")  # this takes inputs without a name !!!
    parser.add_argument('-d', '--data_dir', type=str, default='/well/rittscher/projects/TCGA_prostate/TCGA')
    opt, unknown = parser.parse_known_args()








