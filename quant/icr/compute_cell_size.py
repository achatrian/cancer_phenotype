import argparse
from pathlib import Path
from data import read_annotations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()
    contour_struct = read_annotations(args.data_dir)
    # TODO test
    for slide_id in contour_struct:
        tumour_area_contours = contour_struct[slide_id]['Tumour area']
        # TODO finish







