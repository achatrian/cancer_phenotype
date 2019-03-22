import numpy as np
import cv2
from skimage import measure
from . import read_annotations


def get_region_properties(mask, opencv=True, contour=None):
    """Region props, take the ones that are useful"""
    all_rp = measure.regionprops(mask.astype(np.int32), coordinates='rc')

    if opencv and contour:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        extent = float(area) / rect_area

    for rp in all_rp:
        yield {
            'hu_moments': cv2.HuMoments(cv2.moments(mask)) if opencv else rp.moments_hu,
            'eccentricity': rp.eccentricity,
            'solidity': rp.solidity,
            'extent': extent if opencv and contour else rp.extent,
            'inertia_tensor_eigvals': rp.inertia_tensor_eigvals
        }


def two_layer_region_properties(mask, hier=(200, 250)):
    """ Works for 2 values - 0 is for background
    :param mask:
    :param hier: listing values from the outer contour inwards
    :return:
    """
    assert len(hier) == 2
    if np.unique(mask) == 3:
        mask[mask == hier[1]] = 0
        assert tuple(np.unique(mask)) == (0, hier[1])
        return measure.regionprops(mask)
    else:
        return get_region_properties(mask)


if __name__ == '__main__':
    """Example usage here"""
    #contours = read_annotations()
    pass











