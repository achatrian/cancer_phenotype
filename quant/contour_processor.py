from typing import Sequence, Union
from numbers import Real
import numpy as np
from pandas import DataFrame
from scipy.spatial import distance
from skimage import color
import cv2
from tqdm import tqdm

from data.contours.instance_masker import InstanceMasker
from quant.features import Feature
from data.images.wsi_reader import WSIReader
from data.images.dzi_io import DZIIO
from data.contours import get_contour_image


class ContourProcessor:
    r"""Use this class to iterate over all contours in one annotation"""

    def __init__(self, instance_masker: InstanceMasker, reader: Union[WSIReader, DZIIO], features: Sequence[Feature],
                 contour_size_threshold=2000):
        self.masker = instance_masker
        self.reader = reader
        # check that features have at least one of the input types
        sufficient_args = {'contour', 'mask', 'images', 'gray_image'}
        description = []
        for feature in features:
            if feature.type_.isdisjoint(sufficient_args):
                raise ValueError(f"Feature '{feature.name}' does not contain any of the required input args (may be incomplete)")
            description.extend(feature.returns)
        self.description = description  # what features does the processor output
        self.features = features
        self.contour_size_threshold = contour_size_threshold

    def __len__(self):
        return len(self.masker)

    def __getitem__(self, index):
        mask, components = self.masker[index]
        outer_contour, label = components['parent_contour'], components['parent_label']
        image = get_contour_image(outer_contour, self.reader)
        if cv2.contourArea(outer_contour) < self.contour_size_threshold:
            return None, None, None
        if not self.reader.is_HnE(image, small_obj_size_factor=1/6):
            return None, None, None
        gray_image = color.rgb2gray(image.astype(np.float)).astype(np.uint8)
        # compute features
        features = []
        for feature in self.features:
            kwargs = {}
            if 'contour' in feature.type_:
                kwargs['contour'] = outer_contour
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

        contour_moments = cv2.moments(outer_contour)
        centroid = (round(contour_moments['m10'] / contour_moments['m00']),
                    round(contour_moments['m01'] / contour_moments['m00']))
        data = {
            'index': index,
            'contour': outer_contour.tolist(),  # cannot serialize np.ndarray
            'label': label,
            'centroid': centroid,
            'bounding_rect': cv2.boundingRect(outer_contour)
        }
        return features, self.description, data

    def get_features(self, max_num_features=np.inf):
        r"""Extract features from all contours.
        Also returns small data items about each gland and distance between glands"""
        f, data, bounding_rects, centroids = [], [], [], []
        for feats, self.description, datum in tqdm(self):
            if feats is None:
                continue
            f.append(feats)
            data.append(datum)
            bounding_rects.append(datum['bounding_rect'])
            centroids.append(datum['centroid'])
        f = np.array(f)
        assert f.size > 0, "Feature array must not be empty."
        df = DataFrame(np.array(f),
                       index=tuple('{}_{}_{}_{}'.format(*bb) for bb in bounding_rects),
                       columns=self.description)
        centroids = np.array(centroids)
        dist = distance.cdist(centroids, centroids, 'euclidean')
        return df, data, dist






