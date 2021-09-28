from pathlib import Path
import json
import re
import warnings
import numpy as np
import cv2
from data.images.dzi_io.tile_generator import TileGenerator
from options.process_dzi_options import ProcessDZIOptions
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter


if __name__ == '__main__':
    options = ProcessDZIOptions()
    options.parser.set_defaults(gpu_ids=-1)  # no gpu needed for this script
    opt = options.parse()
    dzi_dir = Path(opt.data_dir)/'data'/'dzi'
    source_file = re.sub('\.(ndpi|svs)', '.dzi', opt.slide_id)
    source_file = source_file if source_file.endswith('.dzi') else source_file + '.dzi'
    target_file = Path('masks')/('mask_' + Path(source_file).name)
    original_dzi = TileGenerator(str(dzi_dir/source_file))  # NB target is used as source !!
    mask_dzi = TileGenerator(str(dzi_dir/target_file))  # NB target is used as source !!
    original_dzi.properties['mpp'], mask_dzi.properties['mpp'] = (float(original_dzi.properties['mpp']),
                                                                  float(mask_dzi.properties['mpp']))
    original_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) *
                                               original_dzi.properties['mpp'] - opt.mpp)))
    converted_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) *
                                                mask_dzi.properties['mpp'] - opt.mpp)))  # relative to lev
    rescale_factor = mask_dzi.properties['mpp'] / original_dzi.properties['mpp']  # to original images
    slide_id = Path(source_file).name[:-4]  # remove .dzi extension
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
    area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3),
                       key=cv2.contourArea)
    # read downsampled region corresponding to tumour area annotation and extract contours
    x, y, w, h = cv2.boundingRect(area_contour)
    # if base layer is not copied from mask, need to read at half the origin as mask dimensions will be halved
    x_read = x if mask_dzi.width == original_dzi.width else int(x / rescale_factor)
    y_read = y if mask_dzi.height == original_dzi.height else int(y / rescale_factor)
    w_read, h_read = int(w / rescale_factor), int(h / rescale_factor)
    mask = mask_dzi.read_region((x_read, y_read), converted_level, (w_read, h_read), border=0)
    converter = MaskConverter()
    print("Extracting contours from mask ...")
    contours, labels, boxes = converter.mask_to_contour(mask, x_read, y_read, rescale_factor=None)  # don't rescale map inside
    # rescale contours
    if rescale_factor != 1.0:
        print(f"Rescaling contours by {rescale_factor:.2f}")
        contours = [(contour * rescale_factor).astype(np.int32) for contour in contours]
    layers = tuple(set(labels))
    print("Storing contours into annotation ...")
    annotation = AnnotationBuilder(opt.slide_id, 'extract_contours', layers)
    for contour, label in zip(contours, labels):
        annotation.add_item(label, 'path')
        contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
        annotation.add_segments_to_last_item(contour)
    if len(annotation) == 0:
        warnings.warn(f"No contours were extracted for slide: {opt.slide_id}")
    annotation.shrink_paths(0.1)
    annotation.add_data('experiment', opt.experiment_name)
    annotation.add_data('load_epoch', opt.load_epoch)
    annotation_dir = Path(opt.data_dir) / 'data' / 'annotations' / opt.experiment_name
    annotation_dir.mkdir(exist_ok=True, parents=True)
    annotation.dump_to_json(annotation_dir)
    print(f"Annotation saved in {str(annotation_dir)}")
    print("Done !")


    


