import re
from pathlib import Path
import pytest
import imageio
from annotation.mask_converter import MaskConverter
from annotation.annotation_builder import AnnotationBuilder


@pytest.fixture
def mask():
    mask_path = '/Volumes/A-CH-EXDISK/Projects/Dataset/train/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,30774,15012,3897,4556)/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,30774,15012,3897,4556)_mask.png'
    mask = imageio.imread(mask_path)
    return mask


@pytest.fixture
def tile_paths():
    tile_dir_path = '/Volumes/A-CH-EXDISK/Projects/Dataset/train/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,30774,15012,3897,4556)/tiles'
    return list(str(path) for path in Path(tile_dir_path).iterdir() if '_mask_' in str(path))


def test_mask_converter(mask):
    converter = MaskConverter()
    converter.by_overlap = False
    contours, labels, boxes = converter.mask_to_contour(mask)
    assert contours


def test_annotation_builder(mask):
    converter = MaskConverter()
    converter.by_overlap = False
    contours, labels, boxes = converter.mask_to_contour(mask)
    aida_ann = AnnotationBuilder('17_A047-4463_153D+-+2017-05-11+09.40.22.ndpi', 'test',
                                 ['epithelium', 'lumen', 'background'])
    aida_ann_dir = '/Users/andreachatrian/Documents/Repositories/AIDA/dist/data/annotations'
    for contour, label, box in zip(contours, labels, boxes):
        aida_ann.add_item(label, 'path', tile_rect=box)
        contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
        aida_ann.add_segments_to_last_item(contour)
    aida_ann.merge_overlapping_segments()
    aida_ann.dump_to_json(aida_ann_dir)


def test_tile_merging(tile_paths):
    converter = MaskConverter()
    aida_ann = AnnotationBuilder('17_A047-4463_153D+-+2017-05-11+09.40.22.ndpi', 'test',
                                 ['epithelium', 'lumen', 'background'])
    # extract offset coords from tile name
    coords_pattern = '\((\w\.\w{1,3}),(\w{1,6}),(\w{1,6}),(\w{1,6}),(\w{1,6})\)_mask_(\w{1,6}),(\w{1,6})'
    for tile_path in tile_paths[0:1]:
        tile = imageio.imread(tile_path)
        coords_info = re.search(coords_pattern, Path(tile_path).name).groups()  # tuple with all matched groups
        downsample = float(coords_info[0])  # downsample is a float
        area_x, area_y, area_w, area_h, tile_x, tile_y = tuple(int(num) for num in coords_info[1:])
        x_offset = area_x + tile_x
        y_offset = area_y + tile_y
        contours, labels, boxes = converter.mask_to_contour(tile, x_offset, y_offset)
        for contour, label, box in zip(contours, labels, boxes):
            aida_ann.add_item(label, 'path', tile_rect=box)
            contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
            aida_ann.add_segments_to_last_item(contour)
    aida_ann_dir = '/Users/andreachatrian/Documents/Repositories/AIDA/dist/data/annotations'
    aida_ann.merge_overlapping_segments(dissimilarity_thresh=20.0, max_iter=5)
    aida_ann.dump_to_json(aida_ann_dir)

#len(tuple(point_dist[p0].keys()))
#next(i for i, (pk, pp) in enumerate(zip(list(point_dist[p0].keys()),points1)) if pk != pp)