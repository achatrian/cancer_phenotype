from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from pytest import fixture
from contours import read_annotations, contour_to_mask, find_overlap, contours_to_multilabel_masks
from quant.contour_processor_ import ContourProcessor
from quant.analyse import quantify
from quant.features import region_properties, red_haralick, green_haralick, blue_haralick, gray_haralick, \
    surf_points, gray_cooccurrence
from data.images.wsi_reader import WSIReader

# fixtures


@fixture
def slide_id():
    return '17_A047-4463_153D+-+2017-05-11+09.40.22'


@fixture
def annotations_dir():
    return Path('/home/andrea/Documents/Repositories/AIDA/dist')


@fixture
def contour_struct(slide_id, annotations_dir):  # fixtures can use other fixtures !
    return read_annotations(annotations_dir, (slide_id,))


@fixture
def contour_lib(contour_struct):
    return contour_struct['17_A047-4463_153D+-+2017-05-11+09.40.22_premerge']


@fixture
def overlap_struct(contour_lib):
    overlap_struct, contours, contour_bbs, labels = find_overlap(contour_lib)
    return overlap_struct


@fixture
def contour_bbs(contour_lib):
    overlap_struct, contours, contour_bbs, labels = find_overlap(contour_lib)
    return contour_bbs


@fixture
def label_values():
    return {'epithelium': 200, 'lumen': 250}


@fixture
def multilabel_mask(contour_lib, overlap_struct, label_values):
    i = next(j for j, overlap_vect in enumerate(overlap_struct) if any(overlap_vect))
    label_values = {'epithelium': 200, 'lumen': 250, 'background': 0}
    masks_gen = contours_to_multilabel_masks(contour_lib, overlap_struct, contour_bbs, label_values, indices=[i])
    return next(masks_gen)  # index was given above, so contour should be the desired one.


@fixture
def reader():
    opt = WSIReader.get_reader_options()
    slide_path = Path('/home/andrea/Documents/Temp/Data/17_A047-4463_153D+-+2017-05-11+09.40.22.ndpi')
    return WSIReader(slide_path, opt)


# tests


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


def test_multilabel_masks(multilabel_mask):
    plt.imshow(multilabel_mask)
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


def test_contour_iterator(contour_lib, overlap_struct, contour_bbs, label_values, reader):
    processor = ContourProcessor(contour_lib, overlap_struct, contour_bbs, label_values, reader,
                                 features=[
                                     region_properties,
                                     red_haralick,
                                     green_haralick,
                                     blue_haralick,
                                     gray_haralick,
                                     surf_points,
                                     gray_cooccurrence
                                 ]
                                 )
    data_iter = iter(processor)
    features, description, data = next(data_iter)


def test_quantify():
    sys.path.append('--data_dir=/home/andrea/Documents/Temp/Data/WSI')
    quantify()
