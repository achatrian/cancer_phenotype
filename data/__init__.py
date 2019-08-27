from pathlib import Path
from functools import partial
from typing import Union
import warnings
import numpy as np
import cv2
from skimage import color
from annotation.annotation_builder import AnnotationBuilder
from images.wsi_reader import WSIReader
from images.dzi_io.dzi_io import DZI_IO
from base.utils import debug

r"""Functions to extract and order images and annotation data, for computing features and clustering"""


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


def read_annotations(data_dir, slide_ids=(), full_path=False):
    r"""Read annotations for one / many slides
    :param data_dir: folder containing the annotation files
    :param slide_ids: ids of the annotations to be read
    :param full_path: if true, the function does not look for the /data/annotations subdir
    :return: dict: annotation id --> (dict: layer_name --> layer points)
    If there are more annotations for the same slide id, these are listed as keys
    """
    assert type(slide_ids) in (tuple, list, set)
    slide_ids = set(slide_ids)
    annotation_dir = Path(data_dir)/'data'/'annotations' if not full_path else Path(data_dir)
    if slide_ids:
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()
                            if any(slide_id in str(annotation_path.name) for slide_id in slide_ids)]
    else:
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()
                            if annotation_path.is_file() and annotation_path.suffix == '.json']
    contour_struct = dict()
    for annotation_path in annotation_paths:
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        annotation_id = annotation_path.name.replace('.json', '')
        contour_struct[annotation_id] = dict()
        for layer_name in annotation.layer_names:
            contour_struct[annotation_id][layer_name], _ = annotation.get_layer_points(layer_name, contour_format=True)
    return contour_struct


def find_overlap(slide_contours: dict, different_labels=True):
    r"""Returns structure detailing which contours overlap, so that overlapping features can be computed
    :param slide_contours:
    :param different_labels: whether overlap is reported only for contours of different classes
    :return:
    """
    contours, labels = [], []
    for layer_name, layer_contours in slide_contours.items():  # merge all layers into one list of contours
        indices = tuple(i for i, contour in enumerate(layer_contours) if contour.size > 2)
        contours.extend(layer_contours[i] for i in indices)
        labels.extend([layer_name] * len(indices))
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
    return overlap_struct, contours, contour_bbs, labels


def contour_to_mask(contour: np.ndarray, value=250, shape=None, mask=None, mask_origin=None):
    r"""Convert a contour to the corresponding mask
    :param contour:
    :param value:
    :param shape: shape of output mask. If not given, mask is as large as contour
    :param pad: amount
    :param mask: mask onto which to paint the new contour
    :param mask_origin: position of mask in slide coordinates
    :param fit_to_size: whether to crop contour to mask if mask is too small or mask to contour if contour is too big
    :return:
    """
    assert type(contour) is np.ndarray and contour.size > 0, "Non-empty numpy array expected for contour"
    #assert fit_to_size in ('mask', 'contour'), "Invalid value for fit_to_size: " + str(fit_to_size)
    contour = contour.squeeze()
    contour_origin = contour.min(0)
    if mask_origin:
        assert len(mask_origin) == 2, "Must be x and y coordinate of mask offset"
        contour = contour - np.array(mask_origin)
        # remove negative points from contour
    else:
        contour = contour - contour_origin  # remove slide offset (don't modify internal reference)
    # below: dimensions of contour to images coords (+1's are to match bounding box dims from cv2.boundingRect)
    contour_dims = (
        contour[:, 1].max() + 1 - contour[:, 1].min(),  # min is not necessarily 0, if mask origin is subtracted instead of min above
        contour[:, 0].max() + 1 - contour[:, 0].min()
    )  # xy to row-columns (rc) coordinates
    if shape is None:  # if shape is not passed, but mask is passed, shape is mask shape. If mask is not passed, shape is as big as contour
        shape = mask.shape if isinstance(mask, np.ndarray) else contour_dims
    if mask is None:
        mask = np.zeros(shape)
    contour = np.clip(contour, (0, 0), (shape[1], shape[0]))  # project all points outside of contour to contour border
    # recompute after removing points in order to test that contour fits in mask
    contour_dims = (
        contour[:, 1].max() + 1 - contour[:, 1].min(),
        contour[:, 0].max() + 1 - contour[:, 0].min()
    )
    assert mask.shape[0] >= contour_dims[0] - 1 and mask.shape[1] >= contour_dims[1] - 1, "Shifted contour should fit in mask"
    cv2.drawContours(mask, [contour], -1, value, thickness=-1)  # thickness=-1 fills the entire area inside
    # assert np.unique(mask).size > 1, "Cannot return empty (0) mask after contour drawing"
    if np.unique(mask).size <= 1:
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # print this warning each time it occurs
            warnings.warn("Returning empty mask after drawing ...")
    return mask


def mark_point_on_mask(mask, point, bounding_box, value=50, radius=0):
    r"""Point is in same coordinates as bounding box"""
    assert mask.dtype == np.uint8
    x, y, w, h = bounding_box
    assert x <= point[0] <= x + w and y <= point[1] <= y + h
    shifted_point = (point[0] - x, point[1] - y)
    if not radius:
        radius = int(np.sqrt(w*h) / 10)  # small w.r.to images size
    radius = max(radius, 1)
    # modifies passed images (no return); negative thickness means fill circle
    cv2.circle(mask, shifted_point, radius, color=value ,thickness=-1)
    return mask


def contours_to_multilabel_masks(slide_contours: Union[dict, tuple], overlap_struct: list, bounding_boxes: list,
                                 label_values: dict, shape=(), contour_to_mask=contour_to_mask, indices=()):
    r"""
    Translate contours into masks, painting masks with overlapping contours with different labels
    :param slide_contours: dict of lists of np.array or (contours,
    :param overlap_struct: overlap between contours
    :param bounding_boxes:
    :param label_values: mapping of contour labels (layer names in slide_contours) to values to be painted on mask
    :param shape: fix shape of output masks (independent on contour size)
    :param contour_to_mask: function for making masks
    :param indices: subset of indices to loop over
    :return: outer contour, mask
    """
    if shape:
        contour_to_mask = partial(contour_to_mask, shape=shape)
    if isinstance(slide_contours, dict):
        contours, labels = [], []
        for layer_name, layer_contours in slide_contours.items():
            contours.extend(layer_contours)
            labels.extend([layer_name] * len(layer_contours))
    else:
        contours, labels = slide_contours
    skips = []  # store contours that have been painted onto other map, and hence should not be returned
    for i in indices or range(len(overlap_struct)):
        if i in skips:
            continue
        if contours[i].size < 2:
            skips.append(i)
            continue
        overlap_vect = overlap_struct[i]
        mask = contour_to_mask(contours[i], value=label_values[labels[i]])
        x_parent, y_parent, w_parent, h_parent = bounding_boxes[i]
        for child_index in np.where(overlap_vect)[0]:
            if contours[child_index].size < 2:
                skips.append(child_index)
            if child_index in skips:
                continue
            x_child, y_child, w_child, h_child = bounding_boxes[child_index]
            if h_parent * w_parent > h_child * w_child:  # if parent bigger than child write on previous mask
                mask = contour_to_mask(contours[child_index], mask=mask, mask_origin=(x_parent, y_parent),
                                       value=label_values[labels[child_index]])
            skips.append(child_index)  # don't yield this contour again
        yield mask, i


def get_contour_image(contour: np.array, reader: Union[WSIReader, DZI_IO], min_size=None):
    r"""
    Extract images from slide corresponding to region covered by contour
    :param contour: area of interest
    :param reader: object implementing .read_region to extract the desired images
    :param min_size:
    """
    x, y, w, h = cv2.boundingRect(contour)
    if min_size is not None:
        assert len(min_size) == 2, "Tuple must contain x and y side lengths of bounding box"
        if w < min_size[0]:
            # x -= min_size[0] // 2 # would break correspondence with mask
            w = min_size[0]
        if h < min_size[1]:
            # y -= min_size[1] // 2
            h = min_size[1]
    # level below: annotation coordinates should refer to lowest level
    image = np.array(reader.read_region((x, y), level=0, size=(w, h)))
    if image.shape[2] == 4:
        image = (color.rgba2rgb(image) * 255).astype(np.uint8)  # RGBA to RGB TODO this failed feature.is_image() test
    return image

# if too slow, could get images for contours using multiprocessing dataloader-style