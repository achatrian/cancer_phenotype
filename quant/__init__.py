from pathlib import Path
from functools import partial
import json
from itertools import chain, combinations
from collections import namedtuple
import numpy as np
import cv2
from base.utils.annotation_builder import AnnotationBuilder
from base.data.wsi_reader import WSIReader

NO_PYTHON = False  # switches from numpy-only to opencv + skimage

r"""Functions to extract and order image and annotation data, for computing features and clustering"""


def annotations_summary(contour_struct, print_file=''):
    r"""Print summary of annotation data.
    :param: output of read_annotation()
    """
    message = ""
    for annotation_id, layer_struct in contour_struct.items():
        slide_message = f"{annotation_id}:"
        for layer_name, contours in layer_struct.items():
            slide_message += f" {layer_name} {len(contours)} contours |"
        message += slide_message + '\n'
    print(message)
    if print_file:
        with open(print_file, 'w') as print_file:
            print(message, file=print_file)


def read_annotations(data_dir, slide_ids=()):
    r"""Read annotations for one / many slides
    :param data_dir: folder containing the annotation files
    :param slide_ids: ids of the annotations to be read
    :return: dict: annotation id --> (dict: layer_name --> layer points)
    If there are more annotations for the same slide id, these are listed as keys
    """
    assert type(slide_ids) in (tuple, list, set)
    slide_ids = set(slide_ids)
    annotation_dir = Path(data_dir)/'data'/'annotations'
    if slide_ids:
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()
                            if any(slide_id in str(annotation_path.name) for slide_id in slide_ids)]
    else:
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()]
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
    for parent_bb, parent_label in zip(contour_bbs, labels):
        # Find out each contour's children (overlapping); if a parent is encountered, no relationship is recorded
        overlap_vector = []
        for child_bb, child_label in zip(contour_bbs, labels):
            if different_labels and parent_label == child_label:
                overlap_vector.append(False)
                continue  # contours of the same class are not classified as overlapping
            position, origin_bb, bb_areas = AnnotationBuilder.check_relative_rect_positions(parent_bb, child_bb, eps=0)
            if parent_bb == child_bb:  # don't do anything for same box
                overlap_vector.append(False)
            elif position == 'contained' and origin_bb == 0:  # flag contour when parent contains child
                overlap_vector.append(True)
            elif position == 'overlap':  # flag contour when parent and child overlap to some extent
                overlap_vector.append(True)
            else:
                overlap_vector.append(False)
        overlap_struct.append(overlap_vector)
    return overlap_struct


def contour_to_mask(contour: np.ndarray, value=250, shape=(), mask=None, hier=(0, 200, 250)):
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
    contour = contour - contour.min(0)  # remove slide offset
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


def contours_to_multilabel_masks(slide_contours: dict, overlap_struct: list, shape=(),
                                 hier=(0, 200, 250), contour_to_mask=contour_to_mask):
    r"""
    Translate contours into masks, painting masks with overlapping contours with different labels
    :param slide_contours: dict of lists of np.array
    :param overlap_struct: overlap between contours
    :param shape: fix shape of output masks (independent on contour size)
    :param contour_to_mask: function for making masks
    :return: outer contour, mask
    """
    if shape or hier:
        contour_to_mask = partial(contour_to_mask, shape=shape, hier=hier)
    masks = []
    contours, labels = [], []
    for layer_name, layer_contours in slide_contours.items():
        contours.extend(layer_contours)
        labels.extend([layer_name] * len(layer_contours))
    for i, overlap_vect in enumerate(overlap_struct):
        mask = contour_to_mask(contours[i])
        for child_idx in np.where(overlap_vect)[0]:
            mask = contour_to_mask(contours[child_idx])  # writes on previous mask
        yield mask


def get_image_for_contour(contour, reader):
    r"""
    Extract image from slide corresponding to region covered by contour
    """
    x, y, w, h = cv2.boundingRect(contour)
    return reader.read_region((x, y), level=0, size=(w, h))  # annotation coordinates should refer to lowest level


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


Feature = namedtuple('Feature', ('type', 'fn'), defaults=((), lambda x: x))  # class used in features to define feature functions


class ContourProcessor:

    def __init__(self, contour_lib, overlap_struct, reader, features=None):
        self.contour_lib = contour_lib
        self.overlap_struct = overlap_struct
        self.features = features
        self.reader = reader
        if features:
            allowed_feature_types = set(powerset(('contour', 'mask', 'image')))
            assert set(feature.type for feature in features) <= allowed_feature_types, "must work on combination of the 3 datatypes"

    def __iter__(self, layer):
        self.gen = contours_to_multilabel_masks(self.contour_lib, self.overlap_struct)
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        outer_contour, mask = next(self.gen)  # will raise stop iteration when no more data is available
        image = get_image_for_contour(outer_contour, self.reader)
        return outer_contour, mask, image




# if too slow, could get images for contours using multiprocessing dataloader-style









