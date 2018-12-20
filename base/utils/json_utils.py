"""
Utility functions for translating binary masks into vertice mappings.
Supports:
PaperJS Path format
QuPath annotations object
"""

import numpy as np
import cv2
import json
import pytest
import imageio
import matplotlib.pyplot as plt
from base.utils import utils


def save_annotation_to_json(filename, mask, x_offset, y_offset, annotation='qupath'):
    assert annotation in ['qupath', 'paperjs']
    contours = get_mask_contours(mask, x_offset, y_offset)
    if annotation == 'paperjs':
        path_emul = PaperJSPathEmulator
    elif annotation == 'qupath':
        path_emul = QuPathAnnotationEmulator
    else:
        raise NotImplementedError(f"Annotation type '{annotation}' is not supported (Supported: ['qupath', paperjs']")
    path_array = [path_emul(contour, None) for contour in contours]

    with open(filename, 'w') as path_file:
        json.dump(path_array, indent=1, fp=path_file, default=path_emul.encode_path_array)


def get_mask_contours(mask, x_offset, y_offset, dist_threshold=0.1):
    """
    :param mask: binary image
    :param x_offset: x_offset offset
    :param y_offset: y_offset offset
    :param dist_threshold: threshold in object core determination using distance transform
    :return: contours of objects in image
    """
    if mask.ndim == 3:
        mask1 = mask[..., 0]
    else:
        mask1 = mask
        mask = np.tile(mask, (1, 1, 3))
    mask = mask if mask.ndim == 3 else mask.tile
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=2)
    # refine background area
    refined_bg = cv2.dilate(opening, kernel, iterations=3)
    # refine foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, refined_fg = cv2.threshold(dist_transform, dist_threshold*dist_transform.max(), 255, 0)
    refined_fg = np.uint8(refined_fg)
    # finding unknown region
    unknown = cv2.subtract(refined_bg, refined_fg)
    # marker labelling
    ret, markers = cv2.connectedComponents(refined_fg)
    markers = markers + 1
    # mark the region of unknown with zero
    markers[unknown == 250] = 0
    # watershed
    markers = cv2.watershed(mask, markers)
    # threshold out boundaries and background (-1 and 0 respectively)
    markers = cv2.morphologyEx(cv2.medianBlur(markers.astype(np.uint8), 3), cv2.MORPH_OPEN, kernel, iterations=2)
    threshold = np.uint8(cv2.medianBlur(markers.astype(np.uint8), 3) > 1) * 255
    # find contours
    im2, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    wsi_frame_contours = []
    for contour in contours:
        contour[..., 0] = contour[..., 0] + x_offset
        contour[..., 1] = contour[..., 1] + y_offset
        wsi_frame_contours.append(contour)
    return wsi_frame_contours


class PaperJSPathEmulator:
    """
    Can use __dict__ property to generate same dict as would paper.Path objects in JS
    """

    @classmethod
    def encode_path_array(cls, path_emulator):
        if isinstance(path_emulator, cls):
            return ['Path', path_emulator.__dict__]
        else:
            raise TypeError(f"Object of type '{path_emulator.__class__.__name__}' is not JSON serializable")

    def __init__(self, contour, label, handles=False, stroke_color=(0, 0, 1)):
        contour = contour.squeeze().tolist()  # work with opencv contour output (numpy array)
        if handles:
            self.segments = [[coords, [0, 0], [0, 0]] for coords in contour]
        else:
            self.segments = [[coords] for coords in contour]
        # attributes in paper.js Path obj
        self.applyMatrix = True
        self.closed = True
        self.strokeColor = list(stroke_color)


class QuPathAnnotationEmulator:

    @classmethod
    def encode_path_array(cls, path_emulator):
        if isinstance(path_emulator, cls):
            return ['Path', path_emulator.__dict__]
        else:
            raise TypeError(f"Object of type '{path_emulator.__class__.__name__}' is not JSON serializable")

    def __init__(self, contour, label):
        contour = contour.squeeze().tolist()


# Tests
@pytest.mark.parametrize('x_offset,y_offset', [(0, 0), (100, 100)])
def test_save_paperjs_path_json(filename, mask, x_offset, y_offset):
    save_annotation_to_json(filename, mask, x_offset, y_offset)


@pytest.mark.parametrize('x_offset,y_offset', [(0, 0), (100, 100)])
def test_get_mask_contours(mask, x_offset, y_offset):
    contours = get_mask_contours(mask, x_offset, y_offset)
    contours_mask = mask.copy()
    cv2.drawContours(contours_mask, contours, -1, 255, 3)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(mask)
    axes[1].imshow(contours_mask)
    plt.show()


@pytest.fixture
def mask():
    mask = imageio.imread('/Users/andreachatrian/Desktop/epoch446_output_map.png')
    mask = np.array(mask)
    return mask


@pytest.fixture
def filename():
    return '/Users/andreachatrian/Desktop/path.json'








