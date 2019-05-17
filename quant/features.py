r"""Feature computations from contours, masks, and images"""

from inspect import getfullargspec
import numpy as np
import cv2
from skimage import measure
from . import read_annotations


class Feature:
    r"""Decorator class to wrap feature functions
    Incorporates type checks on input
    """
    __slots__ = ['function', 'type_', 'name']
    # how-to-python: docstring and __slots__ defined this way are class attributes

    def __init__(self, function):
        # TODO test
        self.function = function
        type_ = set(getfullargspec(function).args)  # get all argument names
        assert type_ >= {'contour'} or type_ >= {'mask'} or type_ >= {'image'}
        self.type_ = type_
        self.name = function.__name__

    @staticmethod
    def is_contour(arg):  # specifies contour format
        return isinstance(arg, np.ndarray) and arg.ndim == 3 and arg.shape[2] == 2

    @staticmethod
    def is_mask(arg, num_classes=0):  # specifies mask format
        return isinstance(arg, np.ndarray) and np.unique(arg) <= (num_classes or 10) and arg.ndim == 2

    @staticmethod
    def is_image(arg):  # specifies image format
        return isinstance(arg, np.ndarray) and arg.max() <= 255 and arg.min() >= 0 and arg.ndim == 3 and arg.shape[2] == 3

    def __call__(self, args, **kwargs):
        if 'contour' in self.type_ and not any(self.is_contour(arg) for arg in args):
            raise ValueError(f"Arguments do not contain contour-type input (f: {self.name})")
        if 'mask' in self.type_ and not any(self.is_mask(arg) for arg in args):
            raise ValueError(f"Arguments does not contain mask-type input (f: {self.name})")
        if 'image' in self.type_ and not any(self.is_mask(arg) for arg in args):
            raise ValueError(f"Arguments does not contain image-type input (f: {self.name})")
        return self.function(args, **kwargs)


@Feature
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


@Feature
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











