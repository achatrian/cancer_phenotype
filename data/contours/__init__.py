from pathlib import Path
from functools import partial
from typing import Union
import warnings
import time
from random import sample
import math
import numpy as np
import cv2
from skimage import color
from openslide import OpenSlideError
from tqdm import tqdm
from rtree import index
from annotation.annotation_builder import AnnotationBuilder
from data.images.wsi_reader import make_wsi_reader, add_reader_args, get_reader_options, WSIReader
from data.images.dzi_io import DZI_IO
from base.utils.timer import Timer
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


def read_annotations(data_dir, slide_ids=(), experiment_name='', full_path=False, annotation_dirname='annotations'):
    r"""Read annotations for one / many slides
    :param annotation_dirname:
    :param experiment_name:
    :param data_dir: folder containing the annotation files
    :param slide_ids: ids of the annotations to be read
    :param full_path: if true, the function does not look for the /data/annotations subdir
    :param annotation_dirname:
    :return: dict: annotation id --> (dict: layer_name --> layer points)
    If there are more annotations for the same slide id, these are listed as keys
    """
    assert type(slide_ids) in (tuple, list, set)
    slide_ids = set(slide_ids)
    if full_path:
        annotation_dir = Path(data_dir)
    else:
        annotation_dir = Path(data_dir)/'data'/annotation_dirname
        if experiment_name:
            annotation_dir = annotation_dir/experiment_name
    # NB path is resolved erroneously in PyCharm (mounted path is used)
    if annotation_dir.is_symlink():
        annotation_dir = annotation_dir.resolve(strict=True)  # resolve symlinks
    if slide_ids:
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()
                            if any(slide_id == annotation_path.stem for slide_id in slide_ids)]
    else:
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()
                            if annotation_path.is_file() and annotation_path.suffix == '.json']
    contour_struct = dict()
    for annotation_path in annotation_paths:
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        annotation_id = annotation_path.name.replace('.json', '')
        contour_struct[annotation_id] = dict()
        for layer_name in annotation.layers:
            contour_struct[annotation_id][layer_name], _ = annotation.get_layer_points(layer_name, contour_format=True)
    if len(annotation_paths) == 0:
        warnings.warn(f"No annotation matched ids: {slide_ids}")
    return contour_struct


def check_relative_rect_positions(tile_rect0, tile_rect1, eps=0):
    r"""
    :param tile_rect0:
    :param tile_rect1:
    :param eps: tolerance in checks (not in contained check !)
    :return: positions: overlap|horizontal|vertical|'' - the relative location of the two paths
             origin_rect: meaning depends on relative positions of two boxes:
                        * contained: which box is bigger
                        * horizontal: leftmost box
                        * vertical: topmost box
                        * overlap: topmost box
                        * '': None
             rect_areas: areas of the bounding boxes
    """
    x0, y0, w0, h0 = tile_rect0
    x1, y1, w1, h1 = tile_rect1
    x_w0, y_h0, x_w1, y_h1 = x0 + w0, y0 + h0, x1 + w1, y1 + h1
    # symmetric relationship - check only in one
    x_overlap = (x0 - eps <= x1 <= x_w0 + eps or x0 - eps <= x_w1 <= x_w0 + eps)  # no need for symmetric check
    y_overlap = (y0 - eps <= y1 <= y_h0 + eps or y0 - eps <= y_h1 <= y_h0 + eps)
    x_contained = (x0 - eps <= x1 <= x_w0 + eps and x0 - eps <= x_w1 <= x_w0 + eps) or \
                  (x1 - eps <= x0 <= x_w1 + eps and x1 - eps <= x_w0 <= x_w1 + eps)  # one is bigger than the other - not symmetric!
    y_contained = (y0 <= y1 <= y_h0 and y0 <= y_h1 <= y_h0) or \
                  (y1 <= y0 <= y_h1 + eps and y1 - eps <= y_h0 <= y_h1)
    if x_contained and y_contained:
        positions = 'contained'
        if (x_w0 - x0) * (y_h0 - y0) >= (x_w1 - x1) * (y_h1 - y1):
            origin_rect = 0  # which box is bigger
        else:
            origin_rect = 1
    elif not x_contained and y_overlap and (x_w0 <= x1 + eps or x_w1 <= x0 + eps):
        positions = 'horizontal'
        if x_w0 <= x1 + eps:
            origin_rect = 0
        elif x_w1 <= x0 + eps:
            origin_rect = 1
        else:
            raise ValueError("shouldn't be here")
    elif not y_contained and x_overlap and (y_h0 <= y1 + eps or y_h1 <= y0 + eps):
        positions = 'vertical'
        if y_h0 <= y1 + eps:
            origin_rect = 0
        elif y_h1 <= y0 + eps:
            origin_rect = 1
        else:
            raise ValueError("shouldn't be here")
    elif x_overlap and y_overlap:
        positions = 'overlap'
        if y0 <= y1:
            origin_rect = 0
        else:
            origin_rect = 1
    else:
        positions = ''
        origin_rect = None
    rect_areas = ((x_w0 - x0) * (y_h0 - y0), (x_w1 - x1) * (y_h1 - y1))
    return positions, origin_rect, rect_areas


def check_point_overlap(parent_contour, child_contour, n_points=5, strict=False):
    r"""Check whether child contour is inside parent contour by testing n_points points for inclusion"""
    # sample child contour
    child_points = sample(child_contour.squeeze().tolist(), min(n_points, child_contour.shape[0]))
    inclusions = tuple(cv2.pointPolygonTest(parent_contour, tuple(point), False) >= 0 for point in child_points)
    return all(inclusions) if strict else any(inclusions)


def find_overlap(slide_contours: dict, different_labels=True, selected_labels=None, verbose=False):
    r"""Returns structure detailing which contours overlap, so that overlapping features can be computed
    :param slide_contours:
    :param different_labels: whether overlap is reported only for contours of different classes
    :return: SPARSE overlap vectors (integers)
    """

    contours, labels = [], []
    for layer_name, layer_contours in slide_contours.items():  # merge all layers into one list of contours
        indices = tuple(i for i, contour in enumerate(layer_contours) if contour.size > 2)
        contours.extend(layer_contours[i] for i in indices)
        labels.extend([layer_name] * len(indices))
    if len(contours) == 0:
        raise ValueError("Empty contour structure")
    contour_bbs = [cv2.boundingRect(contour) for contour in contours]
    start_time = time.time()
    with Timer(None, "Time to construct rtree = {:0.4f}", logger=print if verbose else None):
        idx = index.Rtree(((i, (x, y, x + w, y + h), None) for i, (x, y, w, h) in enumerate(contour_bbs)))
    overlap_struct = []
    boxes_and_labels = zip(contour_bbs, labels)
    if verbose:
        boxes_and_labels = tqdm(boxes_and_labels, total=len(contour_bbs), desc="Discovering overlapping contours ...")
    for parent_bb, parent_label in boxes_and_labels:
        with Timer('overlap', "Time to compute overlap = {:0.4f}", print if verbose else None):
            if selected_labels is not None and parent_label not in selected_labels:
                overlap_struct.append([])
                continue
            x, y, w, h = parent_bb
            all_overlaps = list(idx.intersection((x, y, x + w, y + h)))  # query for overlapping bbs among the tree nodes
            overlap_vector = [i for i in all_overlaps if labels[i] != parent_label] if different_labels else all_overlaps
            overlap_struct.append(overlap_vector)
    if verbose:
        print(f"Average time taken to find overlapping boxes = {Timer.averages['overlap']}")
    return overlap_struct, contours, contour_bbs, labels


def contour_to_mask(contour: np.ndarray, value=250, shape=None, mask=None, mask_origin=None, smoothing=0):
    r"""Convert a contour to the corresponding mask
    :param contour:
    :param value:
    :param shape: shape of output mask. If not given, mask is as large as contour
    :param mask: mask onto which to paint the new contour
    :param mask_origin: position of mask in slide coordinates
    :param smoothing:
    :return:
    """
    # TODO this should always be done relative to original slide in order to avoid confusion
    assert type(contour) is np.ndarray and contour.size > 0, "Non-empty numpy array expected for contour"
    # assert fit_to_size in ('mask', 'contour'), "Invalid value for fit_to_size: " + str(fit_to_size)
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
    if smoothing > 0 and mask.shape[0] > 10 and mask.shape[1] > 10:
        mask_type = mask.dtype
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10)))
        blurred_mask = cv2.blur(mask, (smoothing, smoothing))
        mask = ((blurred_mask > 0.5*value) * value).astype(mask_type)
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
    cv2.circle(mask, shifted_point, radius, color=value, thickness=-1)
    return mask


# DEPRECATED remove - use InstanceMasker instead
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
        for child_index in overlap_vect:
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


def get_contour_image(contour: np.array, reader: WSIReader, min_size=None, level=None, mpp=None,
                      contour_level=0, min_size_enforce='one-sided'):
    r"""
    Extract images from slide corresponding to region covered by contour
    :param level:
    :param mpp:
    :param contour: area of interest
    :param reader: object implementing .read_region to extract the desired images
    :param min_size: minimum size of output image 
    :param contour_level: level of contour coordinates
    :param min_size_enforce: way of expanding read region if min_size is given: add to right side or to both side
    """
    # level below: annotation coordinates should refer to lowest level
    if mpp is not None:
        if level is not None:
            raise ValueError("'level' and 'mpp' cannot be specified at once")
        assert math.isclose(reader.mpp_x, reader.mpp_y, rel_tol=0.0001), "Assuming same resolution in both orientations"
        level = np.argmin(np.absolute(np.array(reader.level_downsamples) * reader.mpp_x - mpp))
    if mpp is not None:  # if mpp is given
        rescale_factor = (reader.mpp_x * reader.level_downsamples[contour_level])/mpp
    else:
        rescale_factor = reader.level_downsamples[contour_level]/reader.level_downsamples[level or 0]
    x, y, w_read, h_read = cv2.boundingRect(contour)
    w_out, h_out = int(w_read * rescale_factor), int(h_read * rescale_factor)
    if min_size is not None:
        assert len(min_size) == 2, "Tuple must contain x and y side lengths of bounding box"
        if min_size_enforce == 'one-sided':
            if w_out < min_size[0]:
                w_read = min(int(min_size[0] / rescale_factor), reader.level_dimensions[contour_level][0] - x)
                w_out = int(w_read * rescale_factor)
            if h_out < min_size[1]:
                h_read = min(int(min_size[1] / rescale_factor), reader.level_dimensions[contour_level][1] - y)
                h_out = int(h_read * rescale_factor)
        elif min_size_enforce == 'two-sided':
            # THIS OPTION BREAKS CORRESPONDENCE WITH MASK WHEN WRITING MANY CONTOURS TO A SINGLE MASK
            # (I.E. ORIGIN CHANGES)
            if w_out < min_size[0]:
                w_ = w_read
                w_read = min(int(min_size[0] / rescale_factor), reader.level_dimensions[contour_level][0] - x)  # at level 0
                x -= max(w_read - w_, 0) // 2  # at level 0 -- would break correspondence with mask
                w_out = int(w_read * rescale_factor)
            if h_out < min_size[1]:
                h_ = h_read
                h_read = min(int(min_size[1] / rescale_factor), reader.level_dimensions[contour_level][1] - y)
                y -= max(h_read - h_, 0) // 2
                h_out = int(h_read * rescale_factor)
        else:
            raise ValueError(f"Invalid read expansion method '{min_size_enforce}'")
    try:
        # WITH SOME IMAGES, READING FROM LEVELS > 0 AT LOCATIONS THAT AREN'T POWERS OF 2 IS NOT RELIABLE.
        # THE SOLUTION IS TO READ AT BASE LEVEL AND DOWNSAMPLE
        # THE IMAGE -- MORE RELIABLE
        # image = np.array(reader.read_region((x, y), level=level or 0, size=(w_read, h_read)))
        image = np.array(reader.read_region((x, y), level=0, size=(w_read, h_read)))
        image = cv2.resize(image.astype(np.uint8), (w_out, h_out))
    except OpenSlideError:
        raise
    if image.shape[2] == 4:
        image = image[..., :3]
    return image


# if too slow, could get images for contours using multiprocessing dataloader-style
