from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pytest import fixture
from quant import read_annotations, contour_to_mask, find_overlap, contours_to_multilabel_masks, annotations_summary
from quant.features import region_properties

@fixture
def slide_id():
    return '17_A047-4463_153D+-+2017-05-11+09.40.22'

@fixture
def annotations_dir():
    return Path('/home/andrea/Documents/Repositories/AIDA/dist')


def test_read_annotations(slide_id, annotations_dir):
    contour_lib = read_annotations((slide_id,), annotations_dir)
    ex_slide_contours = next(iter(contour_lib.values()))
    example = ex_slide_contours['epithelium'][0]
    mask = contour_to_mask(example)
    assert np.unique(mask).size > 1, "Must have both binary values"
    plt.imshow(mask)
    plt.show()
    rp = next(region_properties(mask))
    assert rp


def test_multilabel_masks():
    slide_id = '17_A047-4463_153D+-+2017-05-11+09.40.22'
    annotations_dir = Path('/home/andrea/Documents/Repositories/AIDA/dist')
    contour_struct = read_annotations(annotations_dir, (slide_id,))
    annotations_summary(contour_struct)
    contour_lib = contour_struct['17_A047-4463_153D+-+2017-05-11+09.40.22_premerge']
    overlap_struct, contours, contour_bbs, labels = find_overlap(contour_lib)
    i = next(j for j, overlap_vect in enumerate(overlap_struct) if any(overlap_vect))
    label_values = {'epithelium': 200, 'lumen': 250, 'background': 0}
    masks_gen = contours_to_multilabel_masks(contour_lib, overlap_struct, contour_bbs, label_values, indices=[i])
    example_multilabel_mask = next(masks_gen)  # index was given above, so contour should be the desired one.
    plt.imshow(example_multilabel_mask)  # FIXME still not multilabelled
    plt.show()


def test_features(slide_id, annotations_dir):
    contour_struct = read_annotations(annotations_dir, (slide_id,))
    slide_contours = contour_struct[slide_id]
    overlap_struct = find_overlap(slide_contours)
    masks = contours_to_multilabel_masks(slide_contours, overlap_struct)
    plotted = 0
    for mask in masks:
        if np.unique(mask).size > 2:  # if multilabel (2 + background)
            plt.imshow(mask)
            plotted += 1
        if plotted > 2:
            break


