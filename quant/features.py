r"""Feature computations from contours, masks, and images"""

import numpy as np
import cv2
from skimage import measure
from . import read_annotations, Feature


def region_properties(mask, opencv=True, contour=None):
    """Region props, take the ones that are useful
    :param mask: input mask to compute features from
    :param opencv: whether to compute features using opencv - it can be faster
    :param contour: if given, extracts some features from contour directly
    """
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


def two_layer_region_properties(mask, hier=(200, 250), outer_contour=None):
    """ Works for 2 values - 0 is for background
    :param mask:
    :param hier: listing values from the outer contour inwards
    :param outer_contour
    :return:
    """
    assert len(hier) == 2
    if np.unique(mask) == 3:
        mask[mask == hier[1]] = 0
        assert tuple(np.unique(mask)) == (0, hier[1])
        return measure.regionprops(mask)
    else:
        return region_properties(mask, contour=outer_contour)


region_properties = Feature(('mask', 'contour'), region_properties)
two_layer_region_properties = Feature(('mask', 'contour'), two_layer_region_properties)


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('slide_ids', type=str, nargs='+', help="Slide ids to process")  # this takes inputs without a name !!!
    parser.add_argument('-d', '--data_dir', type=str, default='/well/rittscher/projects/TCGA_prostate/TCGA')
    args, unknown = parser.parse_known_args()
    contour_struct = read_annotations(args.slide_ids, args.data_dir)
    for slide_id in contour_struct:
        pass


if __name__ == '__main__':
    """Example usage here"""
    main()











