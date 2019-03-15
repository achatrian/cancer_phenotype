from pathlib import Path
from quant import read_annotations, contour_to_mask


def test_read_annotations():
    slide_id = '17_A047-4463_153D+-+2017-05-11+09.40.22'
    annotation_path = Path('/home/andrea/Documents/Repositories/AIDA/dist')
    contour_lib = read_annotations((slide_id,), annotation_path)
    ex_slide_contours = next(iter(contour_lib.values()))
    example = ex_slide_contours['epithelium'][0]
    mask = contour_to_mask(example)
