from numbers import Real

import cv2
import numpy as np
import tqdm
from pandas import DataFrame
from scipy.spatial import distance
from skimage import color

from data.__init__ import contours_to_multilabel_masks, get_contour_image


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
        sufficient_args = {'contour', 'mask', 'images', 'gray_image'}
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
        r"""Returns an iterator to go through all the contours in a slide"""
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
        image = get_contour_image(contour, self.reader)
        label = self.labels[index]
        bounding_box = self.bounding_boxes[index]
        if not self.reader.is_HnE(image, small_obj_size_factor=1/6):
            self.discarded.append(index)
            return None, None, None
        self.indices.add(index)
        gray_image = color.rgb2gray(image.astype(np.float)).astype(np.uint8)
        # TODO compute features on resized images too ?
        # scale between 0 and 1
        # images = images / 255.0
        # gray_image = gray_image / 255.0
        # normalise images between -1 and 1
        # images = (images - 0.5) / 0.5
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
