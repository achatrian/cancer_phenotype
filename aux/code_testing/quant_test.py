from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pytest import fixture
from quant import read_annotations, contour_to_mask, find_overlap, contours_to_multilabel_masks
from quant.features import get_region_properties

@fixture
def slide_id():
    return '17_A047-4463_153D+-+2017-05-11+09.40.22'

@fixture
def annotation_path():
    return Path('/home/andrea/Documents/Repositories/AIDA/dist')


def test_read_annotations(slide_id, annotation_path):
    contour_lib = read_annotations((slide_id,), annotation_path)
    ex_slide_contours = next(iter(contour_lib.values()))
    example = ex_slide_contours['epithelium'][0]
    mask = contour_to_mask(example)
    assert np.unique(mask).size > 1, "Must have both binary values"
    plt.imshow(mask)
    plt.show()
    rp = next(get_region_properties(mask))
    assert rp


def test_features(slide_id, annotation_path):
    contour_lib = read_annotations((slide_id,), annotation_path)
    ex_slide_contours = next(iter(contour_lib.values()))
    overlap_struct = find_overlap(ex_slide_contours)
    masks = contours_to_multilabel_masks(ex_slide_contours, overlap_struct)
    plotted = 0
    for mask in masks:
        if np.unique(mask).size > 2:  # if multilabel (2 + background)
            plt.imshow(mask)
            plotted += 1
        if plotted > 2:
            break


