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
from base.data.wsi_reader import WSIReader
from dzi_io.dzi_io import DZI_IO
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
    :param different_labels: whether overlap is reported only for contours of different classes
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
    contour = contour.squeeze()
    if isinstance(mask, np.ndarray) and mask_origin:
        assert len(mask_origin) == 2, "Must be x and y coordinate of mask offset"
        contour = contour - np.array(mask_origin)
    else:
        contour = contour - contour.min(0)  # remove slide offset (don't modify internal reference)
    # below: dimensions of contour to image coords (+1's are to match bounding box dims from cv2.boundingRect)
    contour_dims = (contour[:, 1].max() + 1, contour[:, 0].max() + 1)  # xy to row-columns (rc) coordinates
    shape = mask.shape if not shape and isinstance(mask, np.ndarray) else contour_dims
    if mask is None:
        mask = np.zeros(shape)
    y_diff, x_diff = contour_dims[0] - shape[0], contour_dims[1] - shape[1]
    if fit_to_size == 'contour':
        cut_points = []  # find all the indices of points that would fall outside of mask
        if y_diff > 0:
            cut_points.extend(np.where(contour[:, 1].squeeze() > shape[0])[0])  # y to row
        if x_diff > 0:
            cut_points.extend(np.where(contour[:, 0].squeeze() > shape[1])[0])  # x to column
        points_to_keep = sorted(set(range(contour.shape[0])) - set(cut_points))
        if len(points_to_keep) == 0:
            raise ValueError(f"Contour and mask do not overlap (contour origin {contour.min(0)}, mask shape {shape}, mask origin {mask_origin})")
        contour = contour[points_to_keep, :]
        contour_dims = (contour[:, 1].max() + 1, contour[:, 0].max() + 1)  # xy to rc coordinates
    elif fit_to_size == 'mask':
        pad_width = ((0, y_diff if y_diff > 0 else 0), (0, x_diff if x_diff > 0 else 0), (0, 0))
        mask = np.pad(mask, pad_width, 'constant')
    elif fit_to_size:
        raise ValueError(f"Invalid fit_to_size option: {fit_to_size}")
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


def get_image_for_contour(contour: np.array, reader: Union[WSIReader, DZI_IO]):
    r"""
    Extract image from slide corresponding to region covered by contour
    :param contour: area of interest
    :param reader: object implementing .read_region to extract the desired image
    """
    x, y, w, h = cv2.boundingRect(contour)
    # level below: annotation coordinates should refer to lowest level
    image = reader.read_region((x, y), level=0, size=(w, h)).convert('RGB')
    return np.array(image)


class MultiLabelMasking:
    def __init__(self, slide_contours: Union[dict, tuple], outer_label: str, label_values: dict,
                shape=(), contour_to_mask=contour_to_mask):
        self.slide_contours = slide_contours
        if isinstance(slide_contours, dict):
            self.contours, self.labels = [], []
            for layer_name, layer_contours in slide_contours.items():
                self.contours.extend(layer_contours)
                self.labels.extend([layer_name] * len(layer_contours))
        else:
            self.contours, self.labels = slide_contours
        self.checked_contours_indices = tuple(i for i, contour in enumerate(self.contours) if contour.size < 2)
        self.contours = [contour for i, contour, in enumerate(self.contours) if i in self.checked_contours_indices]
        self.labels = [label for i, label, in enumerate(self.labels) if i in self.checked_contours_indices]
        self.label_values = label_values
        self.outer_label = outer_label
        self.outer_contours_indices = tuple(i for i, label in enumerate(self.labels) if label == outer_label)
        self.overlap_struct, self.contours, self.bounding_boxes, self.labels = find_overlap(slide_contours)
        self.shape = shape
        self.contour_to_mask = partial(contour_to_mask, shape=shape) if shape else contour_to_mask

    def __len__(self):
        return len(self.outer_contours_indices)

    def __getitem__(self, index) -> np.array:
        if index > len(self):
            raise ValueError("Index exceeds number of outer contours")
        outer_index = self.outer_contours_indices[index]
        overlap_vect = self.overlap_struct[outer_index]
        mask = contour_to_mask(self.contours[outer_index], value=self.label_values[self.labels[outer_index]])
        x_parent, y_parent, w_parent, h_parent = self.bounding_boxes[outer_index]
        for child_index in np.where(overlap_vect)[0]:
            x_child, y_child, w_child, h_child = self.bounding_boxes[child_index]
            if h_parent * w_parent > h_child * w_child:  # if parent bigger than child write on previous mask
                mask = contour_to_mask(self.contours[child_index], mask=mask, mask_origin=(x_parent, y_parent),
                                       value=self.label_values[self.labels[child_index]])
        return mask


class ContourProcessor:
    r"""Use this class to iterate over all contours in one annotation
    :param contour_lib: object containing the coordinates of all contours
    :param overlap_struct: list of list (a la matrix) where i,j entry is one if contour i and j overlap
    :param reader: WSIReader instance to read images corresponding to contours
    :param features: list of Features to apply to the contours / images / masks"""
    def __init__(self, contour_lib, overlap_struct, bounding_boxes, label_values, reader, features=()):
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
        self.discarded = []

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
        contour = self.contours[index]
        image = get_image_for_contour(contour, self.reader)
        label = self.labels[index]
        bounding_box = self.bounding_boxes[index]
        if not self.reader.is_HnE(image, small_obj_size_factor=1/6):
            self.discarded.append(index)
            return None, None, None
        self.indices.add(index)
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
            'contour': contour.tolist(),  # cannot serialize np.ndarray
            'label': label,
            'centroid': centroid,
            'bounding_rect': tuple(self.bounding_boxes[index])
        }
        return features, self.description, data

    def get_features(self):
        r"""Extract features from all contours.
        Also returns small data items about each gland and distance between glands"""
        f, data, bounding_rects, centroids = [], [], [], []
        with tqdm.tqdm(total=len(self.contours)) as pbar:
            last_index = 0
            for feats, self.description, datum in self:
                if feats is None:
                    continue
                f.append(feats)
                data.append(datum)
                bounding_rects.append(datum['bounding_rect'])
                centroids.append(datum['centroid'])
                pbar.update(datum['index'] - last_index)
                last_index = datum['index']
        f = np.array(f)
        assert f.size > 0, "Feature array must not be empty."
        df = DataFrame(np.array(f),
                       index=tuple('{}_{}_{}_{}'.format(*bb) for bb in bounding_rects),
                       columns=self.description)
        centroids = np.array(centroids)
        dist = distance.cdist(centroids, centroids, 'euclidean')
        return df, data, dist

# if too slow, could get images for contours using multiprocessing dataloader-style
