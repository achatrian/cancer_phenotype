r"""
Class for translating binary masks into vertice mappings.
Supports:
AIDA format
"""


import copy
import numpy as np
from scipy.stats import mode
from scipy.ndimage import morphology
import skimage.morphology
import cv2
from base.utils import utils, debug


def findContours(mask):
    if int(cv2.__version__.split('.')[0]) == 3:
        _, contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # cv2.CHAIN_APPROX_TC89_KCOS)
    return contours, h


class MaskConverter:
    """
    Class used to convert ground truth annotation to paths / contours in different
    """

    def __init__(self, value_hier=((160, 200), 250), label_value_map=None,
                 label_interval_map=None, label_options=None, fix_ambiguity=True):
        """
        :param value_hier:
        """
        self.value_hier = value_hier
        self.fix_ambiguity = fix_ambiguity
        self.by_overlap = False  # option to use bounding boxes to deal with overlapping contours - for deprecated mask_to_contours_all_classes
        self.label_value_map = label_value_map or {
                'epithelium': 200,
                'lumen': 250,
                'background': 0
            }
        self.label_interval_map = label_interval_map or {
            'epithelium': (70, 225),
            'lumen': (225, 255),
            'background': (0, 70)
        }
        self.label_options = label_options or {  # regulated for masks at resolution=1.0mpp
            'epithelium': {
                'initial_opening_size': 3,
                'small_hole_size': 400,
                'small_object_size': 20,
                'dist_threshold': 0.0001,
                'final_closing_size': 10,
                'final_dilation_size': 5
            },
            'lumen': {
                'initial_opening_size': 1,
                'small_hole_size': 400,
                'small_object_size': 20,
                'dist_threshold': 0.0001,
                'final_closing_size': 10,
                'final_dilation_size': 5
            }
        }
        assert set(self.label_value_map.keys()) == set(self.label_interval_map.keys()), 'inconsistent annotation classes'
        assert all(isinstance(t, tuple) and len(t) == 2 and t[0] <= t[1] for t in self.label_interval_map.values())
        self.num_classes = len(self.label_value_map)

    def mask_to_contour(self, mask, x_offset=0, y_offset=0, rescale_factor=None):
        r"""
        Extracts the contours one class at a time
        :param mask:
        :param x_offset:
        :param y_offset:
        :param rescale_factor: rescale images before extracting contours - in case images was shrunk before being fed to segmenation network
        :return:
        """
        mask = utils.tensor2im(mask, segmap=True, num_classes=self.num_classes, visual=False)  # transforms tensors into mask label images
        if rescale_factor:
            mask = cv2.resize(mask.astype(np.uint8), dsize=None, fx=rescale_factor, fy=rescale_factor, interpolation=cv2.INTER_NEAREST)
        contours, labels = [], []
        for label, interval in self.label_interval_map.items():
            if label == 'background':
                continue  # don't extract contours for background
            value_binary_mask = self.threshold_by_interval(mask, interval)
            if self.fix_ambiguity:
                value_binary_mask = self.remove_ambiguity(value_binary_mask, **self.label_options[label])
            else:
                value_binary_mask = value_binary_mask[..., 0]  # 1 channel mask for contour finding

            value_contours, h = findContours(value_binary_mask)
            value = self.label_value_map[label][1] if isinstance(self.label_value_map[label], (tuple, list)) \
                else self.label_value_map[label]
            value_labels = [value] * len(value_contours)
            contours.extend(value_contours)
            labels.extend(value_labels)
            # use RETR_TREE to get full hierarchy (needed for labels)
        assert len(contours) == len(labels)
        good_contours, good_labels, bounding_boxes = [], [], []
        for i, contour in enumerate(contours):
            # filter contours
            is_good = labels[i] is not None and contour.shape[0] > 2
            if is_good:
                x_offset, y_offset = int(x_offset), int(y_offset)
                bounding_box = list(cv2.boundingRect(contour))
                bounding_box[0] += x_offset
                bounding_box[1] += y_offset
                bounding_boxes.append(tuple(bounding_box))
                good_contours.append(contour + np.array((x_offset, y_offset)))  # add offset
                good_labels.append(self.value2label(labels[i]))
        return good_contours, good_labels, bounding_boxes

    def threshold_by_interval(self, mask, interval):
        r"""
        Use label hierarchy to threshold values
        :param interval:
        :return:
        """
        mask = mask.copy()
        for value_level in reversed(self.value_hier):
            try:
                try:
                    lower, upper = next((lower, upper) for lower, upper in self.label_interval_map.values()
                                        if all(lower <= v <= upper for v in value_level))
                    mask[np.logical_and(lower < mask, mask < upper)] = 1
                    if all(interval[0] <= v < interval[1] for v in value_level):
                        break
                except TypeError:
                    lower, upper = next((lower, upper) for lower, upper in self.label_interval_map.values()
                                          if lower <= value_level <= upper)
                    mask[np.logical_and(lower < mask, mask < upper)] = 1
                    if interval[0] <= value_level < interval[1]:
                        break
            except StopIteration:
                raise ValueError(f"Given intervals don't cover hierarchy value {value_level}")
        mask[mask != 1] = 0
        return mask

    @staticmethod
    def remove_ambiguity(mask, initial_opening_size=3, dist_threshold=0.01, small_hole_size=200, small_object_size=1024*0.4, final_closing_size=20,
                         final_dilation_size=2):
        r"""
        Morphologically removes noise in the images and returns solid contours
        :param mask: HxWx3 images with identical channels, or HxW images
        :param initial_opening_size: size of initial opening done to remove noise
        :param dist_threshold: multiplied by mode of peaks in distance transform -- e,g, 0.1 is 1/10 of the average peak
        :param small_object_size: objects smaller than this threshold will be removed from mask
        :param final_closing_size: size of kernel used for closing of holes in large glands
        :param final_dilation_size: size of kernel used for final dilation of mask values
        :return:
        """
        mask = copy.deepcopy(mask)
        if mask.ndim == 3:
            mask_1c = mask[..., 0]  # need to keep original mask as watershed wants 3 channels images
        else:
            mask_1c = mask
            mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
        # debug.show_image(mask_1c, 'mask_1c')
        # noise removal
        kernel = np.ones((initial_opening_size, initial_opening_size), np.uint8)
        opening = cv2.morphologyEx(mask_1c, cv2.MORPH_OPEN, kernel, iterations=2)
        opening = opening.astype(np.uint8)
        # debug.show_image(opening, 'opening')
        # refine -ve area (includes background)
        refined_bg = cv2.dilate(opening, kernel, iterations=3)
        # refine foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        # get mode of maxima in images, to use as reference for threshold (more invariant than absolute max in tile)
        maxima = skimage.morphology.local_maxima(dist_transform, indices=True)
        maxima = np.stack(maxima, axis=0).T
        values_at_maxima = np.array(list(dist_transform[y, x] for y, x in maxima))
        mode_of_maxima = mode(values_at_maxima)[0]
        mode_of_maxima = mode_of_maxima.item(0) if mode_of_maxima.size > 0 else 255
        # threshold using distance transform
        ret, refined_fg = cv2.threshold(dist_transform,
                                        min(
                                            dist_threshold * mode_of_maxima,
                                            dist_threshold * dist_transform.max()
                                        ),  # threshold
                                        255, cv2.THRESH_BINARY)
        # debug.show_image(refined_fg, 'thresholded')
        refined_fg = np.uint8(refined_fg)
        # finding unknown region
        unknown = cv2.subtract(refined_bg, refined_fg)
        # debug.show_image(unknown, 'unknown')
        # marker labelling
        ret, markers = cv2.connectedComponents(refined_fg)
        markers = markers + 1
        # mark the region of unknown with zero
        markers[unknown > 0] = 0
        # watershed
        markers = cv2.watershed(mask.astype(np.uint8), markers)
        # threshold out boundaries and background (-    1 and 0 respectively)
        markers = cv2.morphologyEx(cv2.medianBlur(markers.astype(np.uint8), 3), cv2.MORPH_OPEN, kernel, iterations=2)
        unambiguous = np.uint8(cv2.medianBlur(markers.astype(np.uint8), 3) > 1) * 255
        # debug.show_image(unambiguous, 'unambiguous')
        # filled holes if any in larger objects
        unambiguous = skimage.morphology.remove_small_holes(unambiguous, small_hole_size)
        # debug.show_image(unambiguous, 'filled')
        # remove small objects
        if small_object_size:
            unambiguous = skimage.morphology.remove_small_objects(unambiguous, min_size=small_object_size)
        # correct troughs left at gland boundaries in larger glands using closing
        if final_closing_size:
            unambiguous = skimage.morphology.binary_closing(unambiguous, np.ones((final_closing_size,)*2))
        # dilate to ensure border was not chipped away by foreground selection above
        if final_dilation_size:
            unambiguous = skimage.morphology.binary_dilation(unambiguous, np.ones((final_dilation_size,)*2))
        return unambiguous.astype(np.uint8)

    def value2label(self, value):
        label = None
        for l, (b1, b2) in self.label_interval_map.items():
            if b1 <= value <= b2:
                if not label:
                    label = l
                    bounds = (b1, b2)
                else:
                    raise ValueError(f"Overlapping interval bounds ({b1}, {b2}) for {l} and {label}")
        if not label:
            raise ValueError(f'Value {value} is not withing any interval')
        return label

    def label2value(self, label):
        return self.label_value_map[label]

    @staticmethod
    def check_bounding_boxes_overlap(parent_bb, child_bb):
        x0, y0, w0, h0 = parent_bb
        x1, y1, w1, h1 = child_bb
        x_w0, y_h0, x_w1, y_h1 = x0 + w0, y0 + h0, x1 + w1, y1 + h1
        return (x0 <= x1 <= x_w0 and
                x0 <= x_w1 <= x_w0 and
                y0 <= y1 <= y_h0 and
                y0 <= y_h1 <= y_h0)