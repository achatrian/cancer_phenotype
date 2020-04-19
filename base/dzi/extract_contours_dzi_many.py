from pathlib import Path
import json
import re
import warnings
import numpy as np
import cv2
from tqdm import tqdm
from data.images.dzi_io.tile_generator import TileGenerator
from options.process_dzi_options import ProcessDZIOptions
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter


if __name__ == '__main__':
    options = ProcessDZIOptions()
    options.parser.set_defaults(gpu_ids=-1)  # no gpu needed for this script
    opt = options.parse()
    opt.display_id = -1   # no visdom display
    dzi_dir = Path(opt.data_dir)/'data'/'dzi'
    image_paths = list(path for path in Path(opt.data_dir).iterdir()
                       if path.name.endswith('.svs') or path.name.endswith('.ndpi') or path.name.endswith('.tiff'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.tiff'))
    annotation_dir = Path(opt.data_dir) / 'data' / 'annotations' / opt.experiment_name
    annotation_dir.mkdir(exist_ok=True, parents=True)
    failure_log = []
    for image_path in tqdm(image_paths):
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        if any(path.name.startswith(slide_id) for path in annotation_dir.iterdir() if path.suffix == '.json'):
            continue
        tqdm.write(f"Processing slide: {slide_id}")
        source_file = re.sub(r'\.(ndpi|svs|tiff)', '.dzi', slide_id)
        source_file = source_file if source_file.endswith('.dzi') else source_file + '.dzi'
        target_file = Path('masks')/('mask_' + Path(source_file).name)
        try:
            original_dzi = TileGenerator(str(dzi_dir/source_file))  # NB target is used as source !!
        except FileNotFoundError as err:
            failure_log.append({
                'file': str(dzi_dir/source_file),
                'error': err,
                'message': f"Dzi image file at {str(dzi_dir/source_file)} is missing or incomplete"
            })
            tqdm.write(f"Dzi image file at {str(dzi_dir/source_file)} is missing or incomplete")
            continue  # move onto next slide
        try:
            mask_dzi = TileGenerator(str(dzi_dir/target_file))  # NB target is used as source !!
        except FileNotFoundError as err:
            failure_log.append({
                'file': str(dzi_dir/target_file),
                'error': err,
                'message': f"Dzi image file at {str(dzi_dir / source_file)} is missing or incomplete"
            })
            tqdm.write(f"Dzi segmentation mask file at {str(dzi_dir/source_file)} is missing or incomplete")
            continue
        original_dzi.properties['mpp'], mask_dzi.properties['mpp'] = (float(original_dzi.properties['mpp']),
                                                                      float(mask_dzi.properties['mpp']))
        original_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) *
                                                   original_dzi.properties['mpp'] - opt.mpp)))
        converted_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) *
                                                    mask_dzi.properties['mpp'] - opt.mpp)))  # relative to lev
        rescale_factor = mask_dzi.properties['mpp'] / original_dzi.properties['mpp']  # to original images
        slide_id = Path(source_file).name[:-4]  # remove .dzi extension
        # read tumour area annotation
        try:
            with open(Path(opt.data_dir) / 'data' / opt.area_annotation_dir /
                      (source_file[:-4] + '.json'), 'r') as annotation_file:
                annotation_obj = json.load(annotation_file)
                annotation_obj['slide_id'] = slide_id
                annotation_obj['project_name'] = 'tumour_area'
                annotation_obj['layer_names'] = ['Tumour area']
                contours, layer_name = AnnotationBuilder.from_object(annotation_obj). \
                    get_layer_points('Tumour area', contour_format=True)
        except FileNotFoundError as err:
            failure_log.append({
                'file': str(Path(opt.data_dir) / 'data' / opt.area_annotation_dir / (source_file[:-4] + '.json')),
                'error': str(err),
                'message': f"No tumour area annotation file for {slide_id}"
            })
            continue
        # biggest contour is used to select the area to process
        area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3),
                           key=cv2.contourArea)
        # read downsampled region corresponding to tumour area annotation and extract contours
        x, y, w, h = cv2.boundingRect(area_contour)
        # if base layer is not copied from mask, need to read at half the origin as mask dimensions will be halved
        x_read = x if mask_dzi.width == original_dzi.width else int(x / rescale_factor)
        y_read = y if mask_dzi.height == original_dzi.height else int(y / rescale_factor)
        w_read, h_read = int(w / rescale_factor), int(h / rescale_factor)
        try:
            mask = mask_dzi.read_region((x_read, y_read), converted_level, (w_read, h_read), border=0)
        except FileNotFoundError as err:
            failure_log.append({
                'file': str(dzi_dir / target_file),
                'error': err,
                'message': f"Dzi segmentation mask file at {str(dzi_dir/source_file)} is incomplete (x: {x_read}, y: {y_read}, w: {w_read}, h: {h_read})"
            })
            tqdm.write(f"Dzi segmentation mask file at {str(dzi_dir/source_file)} is incomplete")
            continue
        converter = MaskConverter()
        tqdm.write("Extracting contours from mask ...")
        contours, labels, boxes = converter.mask_to_contour(mask, x_read, y_read, rescale_factor=None)  # don't rescale map inside
        # rescale contours
        if rescale_factor != 1.0:
            tqdm.write(f"Rescaling contours by {rescale_factor:.2f}")
            contours = [(contour * rescale_factor).astype(np.int32) for contour in contours]
        layers = tuple(set(labels))
        tqdm.write("Storing contours into annotation ...")
        annotation = AnnotationBuilder(slide_id, 'extract_contours', layers)
        for contour, label in zip(contours, labels):
            annotation.add_item(label, 'path')
            contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
            annotation.add_segments_to_last_item(contour)
        if len(annotation) == 0:
            warnings.warn(f"No contours were extracted for slide: {slide_id}")
        annotation.shrink_paths(0.1)
        annotation.add_data('experiment', opt.experiment_name)
        annotation.add_data('load_epoch', opt.load_epoch)
        annotation.dump_to_json(annotation_dir)
        tqdm.write(f"Annotation saved in {str(annotation_dir)}")
    with (annotation_dir.parent/'annotation_extraction_failure_log.json').open('w') as failure_log_file:
        json.dump(failure_log, failure_log_file)
    tqdm.write("Done !")



    


