from pathlib import Path
from argparse import ArgumentParser
import json
import numpy as np
import cv2
import tqdm
from image.dzi_io import TileGenerator
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter

r"""Script to run mask extraction for all .dzi's in one process"""
# TODO useful ?


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--mpp', type=float, default=0.5)
    parser.add_argument('--area_annotation_dir', type=str, default='tumour_area_annotations')
    parser.add_argument('--distance_threshold', type=float, default=0.01,
                        help="Value multiplied by peak in distance transform of mask to threshold objects")
    opt = parser.parse_args()
    dzi_dir = Path(opt.data_dir)/'data'/'dzi'
    dzi_paths = list(path for path in dzi_dir.iterdir() if path.is_file() and path.suffix == '.dzi')
    for dzi_path in tqdm.tqdm(dzi_paths):
        slide_id = dzi_path.name[:-4]
        source_file = dzi_path
        target_file = Path('masks')/('mask_' + Path(source_file).name)
        original_dzi = TileGenerator(str(dzi_dir/source_file))  # NB target is used as source !!
        mask_dzi = TileGenerator(str(dzi_dir/target_file))  # NB target is used as source !!
        original_dzi.properties['mpp'], mask_dzi.properties['mpp'] = (float(original_dzi.properties['mpp']),
                                                                      float(mask_dzi.properties['mpp']))
        original_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) *
                                                   original_dzi.properties['mpp'] - opt.mpp)))
        converted_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) *
                                                    mask_dzi.properties['mpp'] - opt.mpp)))  # relative to lev
        rescale_factor = mask_dzi.properties['mpp'] / original_dzi.properties['mpp']  # to original image
        # read tumour area annotation
        with open(Path(opt.data_dir) / 'data' / opt.area_annotation_dir /
                  (source_file[:-4] + '.json'), 'r') as annotation_file:
            annotation_obj = json.load(annotation_file)
            annotation_obj['slide_id'] = slide_id
            annotation_obj['project_name'] = 'tumour_area'
            annotation_obj['layer_names'] = ['Tumour area']
            contours, layer_name = AnnotationBuilder.from_object(annotation_obj). \
                get_layer_points('Tumour area', contour_format=True)
        # biggest contour is used to select the area to process
        area_contour = max((contour for contour in contours), key=cv2.contourArea)
        # read downsampled region corresponding to tumour area annotation and extract contours
        x, y, w, h = cv2.boundingRect(area_contour)
        # if base layer is not copied from mask, need to read at half the origin as mask dimensions will be halved
        x_read = x if mask_dzi.width == original_dzi.width else int(x / rescale_factor)
        y_read = y if mask_dzi.height == original_dzi.height else int(y / rescale_factor)
        w_read, h_read = int(w / rescale_factor), int(h / rescale_factor)
        mask = mask_dzi.read_region((x_read, y_read), converted_level, (w_read, h_read))
        converter = MaskConverter(dist_threshold=opt.distance_threshold)
        tqdm.tqdm.write("Extracting contours from mask ...")
        contours, labels, boxes = converter.mask_to_contour(mask, x_read, y_read, rescale_factor=None)  # don't rescale map inside
        # rescale contours
        if rescale_factor != 1.0:
            tqdm.tqdm.write(f"Rescaling contours by {rescale_factor:.2f}")
            contours = [(contour * rescale_factor).astype(np.int32) for contour in contours]
        layers = tuple(set(labels))
        tqdm.tqdm.write("Storing contours into annotation ...")
        annotation = AnnotationBuilder(slide_id, 'extract_contours', layers)
        for contour, label in zip(contours, labels):
            annotation.add_item(label, 'path')
            contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
            annotation.add_segments_to_last_item(contour)
        annotation.shrink_paths(0.1)
        annotation.dump_to_json(Path(opt.data_dir)/'data'/'annotations')
        tqdm.tqdm.write(f"Annotation saved in {str(Path(opt.data_dir)/'data'/'annotations')}")
    tqdm.tqdm.write("Done !")

    


