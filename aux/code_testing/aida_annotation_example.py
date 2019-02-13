import re
from pathlib import Path
import imageio
from base.utils.annotation_converter import AnnotationConverter
from base.utils.aida_annotation import AIDAnnotation

converter = AnnotationConverter(min_contour_area=50000)
aida_ann = AIDAnnotation('17_A047-4463_153D+-+2017-05-11+09.40.22.ndpi', 'test',
                         ['epithelium', 'lumen', 'background'])
# extract offset coords from tile name
coords_pattern = '\((\w\.\w{1,3}),(\w{1,6}),(\w{1,6}),(\w{1,6}),(\w{1,6})\)_mask_(\w{1,6}),(\w{1,6})'
tile_path0 = '/Volumes/A-CH-EXDISK/Projects/Dataset/train/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,30774,15012,3897,4556)/tiles/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,30774,15012,3897,4556)_mask_1,2049.png'
tile_path1 = '/Volumes/A-CH-EXDISK/Projects/Dataset/train/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,30774,15012,3897,4556)/tiles/17_A047-4463_153D+-+2017-05-11+09.40.22_TissueTrain_(1.00,30774,15012,3897,4556)_mask_628,2508.png'
for tile_path in [tile_path0, tile_path1]:
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