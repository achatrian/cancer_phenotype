from pathlib import Path
from functools import partial
from typing import Union
from numbers import Real
import warnings
import numpy as np
from scipy.spatial import distance
import cv2
from skimage import color
from pandas import DataFrame
import tqdm
from base.utils.annotation_builder import AnnotationBuilder
from base.utils import utils
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
        annotation_paths = [annotation_path for annotation_path in annotation_dir.iterdir()
                            if annotation_path.is_file() and annotation_path.suffix == '.json']
    contour_struct = dict()
    for annotation_path in annotation_paths:
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
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
    for layer_name, layer_contours in slide_contours.items():  # merge all layers into one list of contours
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
    return overlap_struct, contours, contour_bbs, labels


def contour_to_mask(contour: np.ndarray, value=250, shape=(), mask=None, mask_origin=None,
                    fit_to_size='contour'):
    r"""Convert a contour to the corresponding max - mask is
    :param contour:
    :param value:
    :param shape: shape of output mask. If not given, mask is as large as contour
    :param pad: amount
    :param mask: mask onto which to paint the new contour
    :param mask_origin: position of mask in slide coordinates
    :param fit_to_size: whether to crop contour to mask if mask is too small or mask to contour if contour is too big
    :return:
    """
    assert type(contour) is np.ndarray and contour.size > 0, "Numpy array expected for contour"
    assert fit_to_size in ('mask', 'contour'), "Invalid value for fit_to_size: " + str(fit_to_size)
    assert shape or mask is not None or (len(shape) == 3 and shape[2] == 3), "If mask is created from shape, this must be of the form (H, W, 3)"
    if mask is not None and mask.ndim == 2:
        warnings.warn("Contour drawing works only on 3 channel mask, expanding color dim ...")
        mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
    contour = contour.squeeze()
    if isinstance(mask, np.ndarray) and mask_origin:
        assert len(mask_origin) == 2, "Must be x and y coordinate of mask offset"
        contour = contour - np.array(mask_origin)
    else:
        contour = contour - contour.min(0)  # remove slide offset (don't modify internal reference)
    # below: dimensions of contour to image coords (+1's are to match bounding box dims from cv2.boundingRect)
    contour_dims = (contour[:, 1].max() + 1, contour[:, 0].max() + 1)  # xy to rc coordinates
    shape = mask.shape if not shape and isinstance(mask, np.ndarray) else contour_dims
    if mask is None:
        mask = np.zeros(shape)
    y_diff, x_diff = contour_dims[0] - shape[0], contour_dims[1] - shape[1]
    if fit_to_size == 'contour':
        cut_points = []  # find all the indices of points that would fall outside of mask
        if y_diff > 0:
            cut_points.extend(np.where(contour[:, 0].squeeze() > shape[0])[0])
        if x_diff > 0:
            cut_points.extend(np.where(contour[:, 1].squeeze() > shape[1])[0])
        points_to_keep = sorted(set(range(contour.shape[0])) - set(cut_points))
        if len(points_to_keep) == 0:
            raise ValueError(f"Contour and mask do not overlap (contour origin {contour.min(0)}, mask shape {shape}, mask origin {mask_origin})")
        contour = contour[points_to_keep, :]
    elif fit_to_size == 'mask':
        pad_width = ((0, y_diff if y_diff > 0 else 0), (0, x_diff if x_diff > 0 else 0), (0, 0))
        mask = np.pad(mask, pad_width, 'constant')
    assert mask.shape[0] >= contour[:, 1].max() + 1 and mask.shape[1] >= contour[:, 0].max() + 1
    cv2.drawContours(mask, [contour], -1, (value, 0, 0), thickness=-1)  # thickness=-1 fills the entire area inside
    return mask


def mark_point_on_mask(mask, point, bounding_box, value=50, radius=0):
    r"""Point is in same coordinates as bounding box"""
    assert mask.dtype == np.uint8
    x, y, w, h = bounding_box
    assert x <= point[0] <= x + w and y <= point[1] <= y + h
    shifted_point = (point[0] - x, point[1] - y)
    if not radius:
        radius = int(np.sqrt(w*h) / 10)  # small w.r.to image size
    radius = max(radius, 1)
    # modifies passed image (no return); negative thickness means fill circle
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
        for child_idx in np.where(overlap_vect)[0]:
            if contours[child_idx].size < 2:
                skips.append(child_idx)
            if child_idx in skips:
                continue
            x_child, y_child, w_child, h_child = bounding_boxes[child_idx]
            if h_parent * w_parent > h_child * w_child:  # if parent bigger than child write on previous mask
                mask = contour_to_mask(contours[child_idx], mask=mask, mask_origin=(x_parent, y_parent),
                                       value=label_values[labels[child_idx]])
            skips.append(child_idx)  # don't yield this contour again
        yield mask, i


def get_image_for_contour(contour: np.array, reader: WSIReader):
    r"""
    Extract image from slide corresponding to region covered by contour
    """
    x, y, w, h = cv2.boundingRect(contour)
    # level below: annotation coordinates should refer to lowest level
    image = reader.read_region((x, y), level=0, size=(w, h)).convert('RGB')
    return np.array(image)


class ContourProcessor:
    r"""Use this class to iterate over all contours in one annotation
    :param contour_lib: object containing the coordinates of all contours
    :param overlap_struct: list of list (a la matrix) where i,j entry is one if contour i and j overlap
    :param reader: WSIReader instance to read images corresponding to contours
    :param features: list of Features to apply to the contours / images / masks"""
    def __init__(self, contour_lib, overlap_struct, bounding_boxes, label_values, reader, features=()):
        # TODO test
        if isinstance(contour_lib, dict):
            contours, labels = [], []
            for layer_name, layer_contours in contour_lib.items():
                contours.extend(layer_contours)
                labels.extend([layer_name] * len(layer_contours))
        else:
            contours, labels = contour_lib
        self.contours = contours
        self.labels = labels
        self.overlap_struct = overlap_struct
        self.bounding_boxes = bounding_boxes
        self.label_values = label_values
        self.features = features
        self.reader = reader
        # check that features have at least one of the input types
        sufficient_args = {'contour', 'mask', 'image', 'gray_image'}
        description = []
        for feature in features:
            if feature.type_.isdisjoint(sufficient_args):
                raise ValueError(f"Feature '{feature.name}' does not contain any of the required input args (may be incomplete)")
            description.extend(feature.returns)
        self.description = description  # what features does the processor output
        self.skip_label = None
        self.indices = set()

    def __iter__(self, skip_label='lumen'):
        if skip_label:
            self.indices = [i for i in range(len(self.labels)) if self.labels[i] != skip_label]
        self.gen = contours_to_multilabel_masks((self.contours, self.labels), self.overlap_struct,
                                                self.bounding_boxes, self.label_values, indices=self.indices)
        self.i = 0
        self.skip_label = skip_label
        self.indices = set(self.indices)
        return self

    def __next__(self):
        self.i += 1
        mask, index = next(self.gen)  # will raise stop iteration when no more data is available
        self.indices.add(index)
        contour = self.contours[index]
        label = self.labels[index]
        image = get_image_for_contour(contour, self.reader)
        gray_image = color.rgb2gray(image.astype(np.float)).astype(np.uint8)
        # TODO compute features on resized image too ?
        # scale between 0 and 1
        # image = image / 255.0
        # gray_image = gray_image / 255.0
        # normalise image between -1 and 1
        # image = (image - 0.5) / 0.5
        # gray_image = (gray_image - 0.5) / 0.5
        features = []
        for feature in self.features:
            kwargs = {}
            if 'contour' in feature.type_:
                kwargs['contour'] = contour
            if 'mask' in feature.type_:
                kwargs['mask'] = mask
            if 'image' in feature.type_:
                kwargs['image'] = image
            if 'gray_image' in feature.type_:
                kwargs['gray_image'] = gray_image
            if 'label' in feature.type_:
                kwargs['label'] = label
            output = feature(**kwargs)  # each entry of output must be a different feature
            features.extend(output)
            assert all(isinstance(f, Real) for f in output), f"Features must be real numbers (invalid output for {feature.name})"
        contour_moments = cv2.moments(self.contours[index])
        centroid = (round(contour_moments['m10'] / contour_moments['m00']),
                    round(contour_moments['m01'] / contour_moments['m00']))
        data = {
            'index': index,
            'contour': contour,
            'mask': mask,
            'image': image,
            'label': label,
            'centroid': centroid,
            'bounding_rect': self.bounding_boxes[index]
        }
        return features, self.description, data

    def get_features(self):
        r"""Extract features from all contours"""
        f, d, centroids = [], [], []
        with tqdm.tqdm(total=len(self.contours)) as pbar:
            last_index = 0
            for feats, self.description, data in self:
                f.append(feats)
                d.append({
                    'index': data['index'],
                    'label': data['label'],
                    'centroid': data['centroid'],
                    'bounding_rect': data['bounding_rect']
                })  # avoid returning the mask, image and contour data
                centroids.append(data['centroid'])
                pbar.update(data['index'] - last_index)
                last_index = data['index']
        df = DataFrame(np.array(f), index=centroids, columns=self.description)
        centroids = np.array(centroids)
        dist = distance.cdist(centroids, centroids, 'euclidean')
        return df, d, dist

# if too slow, could get images for contours using multiprocessing dataloader-style




