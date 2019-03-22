from pathlib import Path
from functools import partial
import argparse
import json
import numpy as np
import cv2
from numba import jit
from base.utils.annotation_builder import AnnotationBuilder
NO_PYTHON = False  # switches from numpy-only to opencv + skimage


def read_annotations(slide_ids, data_dir):
    r"""Read annotations for one / many slides
    :return: dict: annotation id --> (dict: layer_name --> layer points)
    """
    assert type(slide_ids) in (tuple, list, set)
    slide_ids = set(slide_ids)
    annotation_dir = Path(data_dir)/'data'/'annotations'
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


def find_overlap(slide_contours: dict, different_labels=True):
    r"""Returns structure detailing which contours overlap, so that overlapping features can be computed
    :param slide_contours:
    :param different_labels: o
    :return:
    """
    contours, labels = [], []
    for layer_name, layer_contours in slide_contours.items():
        contours.extend(layer_contours)
        labels.extend([layer_name] * len(layer_contours))
    contour_bbs = list(cv2.boundingRect(contour) for i, contour in enumerate(contours))
    overlap_struct = []
    for parent_bb, parent_label in contour_bbs, labels:
        # Find out each contour's children (overlapping); if a parent is encountered, no relationship is recorded
        overlap_vector = []
        for child_bb, child_label in contour_bbs, labels:
            if different_labels and parent_label == child_label:
                overlap_vector.append(False)
                continue  # contours of the same class are not classified as overlapping
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


def contour_to_mask(contour: np.ndarray, value=255, shape=(), mask=None):
    r"""Convert a contour to the corresponding max - mask is
    :param contour:
    :param value:
    :param shape: shape of output mask. If not given, mask is as large as contour
    :param pad: amount
    :param mask: mask onto which to paint the new contour
    :return:
    """
    assert type(contour) is np.ndarray, "Numpy array expected for contour"
    contour = contour.squeeze()
    contour = contour - contour.min(0)
    contour_shape = (contour[:, 1].max(), contour[:, 0].max())  # dimensions of contour to image coords
    if shape:
        cut_points = []  # find all the indices of points that would fall outside of mask
        if shape[1] < contour_shape[1]:
            cut_points.extend(np.where(contour[:, 1] > shape[0])[0])
        if shape[0] < contour_shape[0]:
            cut_points.extend(np.where(contour[:, 0] > shape[0])[0])
        contour = contour[cut_points, :]
    elif mask:
        shape = mask.shape
    else:
        shape = contour_shape
    mask = mask or np.zeros(shape)
    mask = np.pad(mask, np.array(mask.shape) - np.array(shape), 'constant')
    assert mask.shape == shape
    cv2.drawContours(mask, [contour], -1, value, thickness=-1)  # thickness=-1 fills the entire area inside
    return mask


def contours_to_multilabel_masks(slide_contours: dict, overlap_struct: list, shape=(), contour_to_mask=contour_to_mask):
    r"""
    Translate contours into masks, painting masks with overlapping contours with different labels
    :param slide_contours: dict of lists of np.array
    :param overlap_struct: overlap between contours
    :param shape: fix shape of output masks (independent on contour size)
    :param contour_to_mask: function for making masks
    :return:
    """
    if shape:
        contour_to_mask = partial(contour_to_mask, shape=shape)
    masks = []
    contours, labels = [], []
    for layer_name, layer_contours in slide_contours.items():
        contours.extend(layer_contours)
        labels.extend([layer_name] * len(layer_contours))
    for i, overlap_vect in enumerate(overlap_struct):
        mask = contour_to_mask(contours[i])
        for child_idx in np.where(overlap_vect)[0]:
            mask = contour_to_mask(contours[child_idx], mask)
        masks.append(mask)
    return masks


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slide_ids', type=str, nargs='+', help="Slide ids to process")  # this takes inputs without a name !!!
    parser.add_argument('-d', '--data_dir', type=str, default='/well/rittscher/projects/TCGA_prostate/TCGA')
    opt, unknown = parser.parse_known_args()








