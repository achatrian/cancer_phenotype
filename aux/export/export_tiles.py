import pickle
import sys
import warnings
from pathlib import Path
import json
from os.path import basename as os_basename
sys.path.extend([
    '/well/rittscher/users/achatrian/cancer_phenotype/',
    '/well/rittscher/users/achatrian/cancer_phenotype/base'
])
from base.data.wsi_reader import WSIReader
from base.utils.annotation_builder import AnnotationBuilder
from base.options.base_options import BaseOptions


def main(slide_file):
    sys.argv.pop(0)
    sys.argv.append('--dataset_name=wsi')
    sys.argv.append('--gpu_ids=-1')
    sys.argv.append('--patch_size=3200')
    sys.argv.append('--mpp=0.5')
    sys.argv.append('--data_dir=/well/rittscher/projects/TCGA_prostate/TCGA')
    sys.argv.append('--verbose')
    print(f"Py Processing {os_basename(slide_file)}")
    opt = BaseOptions().parse()
    # Load annotation:
    tumour_annotations_dir = Path(opt.data_dir) / 'data' / 'tumour_area_annotations'
    annotation_path = tumour_annotations_dir / Path(slide_file).with_suffix('.json').name
    with open(annotation_path, 'r') as annotation_file:
        annotation = json.load(annotation_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contours, labels = AnnotationBuilder.from_object(annotation).get_layer_points(0, contour_format=True)
    setattr(opt, 'overwrite_qc', True)  # force overwriting of all quality_control files
    print(f"Quality control mpp: {opt.qc_mpp}, read_mpp: {opt.mpp}")
    slide = WSIReader(opt, slide_file)
    slide.find_good_locations()
    slide.export_good_tiles('tiles2', export_contours=contours, contour_scaling=0.1)


if __name__ == '__main__':
    main(sys.argv[1])  # first entry in argv is name of file


