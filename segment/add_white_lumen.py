import argparse
from pathlib import Path
from data.contours import read_annotations, get_contour_image
from data.contours.instance_masker import InstanceMasker

r"""
Script to add lumen to gland annotations where it is white and easy to single out using handcrafted methods
This should allow retraining of the segmentation algorithm in order to improve the outcome of clustering
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('experiment_name', type=str)
    args = parser.parse_args()

    contour_struct = read_annotations(args.data_dir)
    for slide_id in contour_struct:
        slide_contours = contour_struct[slide_id]
        masker = InstanceMasker(contour_struct, 'epithelium', dict((('epithelium', 200), ('lumen', 250))))
        for mask, components in masker:
            image = get_contour_image(components['parent_contour'])  # TODO finish, must make WSIReader
            pass






