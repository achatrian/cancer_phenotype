import warnings
from pathlib import Path
from functools import partial
import json
import socket
import re
from datetime import datetime
import multiprocessing as mp
import cv2
import torch
import numpy as np
from tqdm import tqdm
from openslide import OpenSlideUnsupportedFormatError
from base.options.process_openslide_options import ProcessOpenSlideOptions
from data.images.wsi_reader import WSIReader
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter
from base.utils import debug


def extract_annotation(image_path, opt, failure_log):
    # read tumour area annotation
    annotation_path = Path(opt.data_dir) / 'data' / 'annotations' / opt.experiment_name / image_path.with_suffix(
        '.json').name
    if annotation_path.exists():
        return None
    mask_path = Path(opt.data_dir) / 'data' / opt.mask_dirname / image_path.with_suffix('.tiff').name
    slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
    try:
        with open(Path(opt.data_dir) / 'data' / opt.area_annotation_dir /
                  (slide_id + '.json'), 'r') as annotation_file:
            annotation_obj = json.load(annotation_file)
            annotation_obj['slide_id'] = slide_id
            annotation_obj['project_name'] = 'tumour_area'
            annotation_obj['layer_names'] = ['Tumour area']
            annotation = AnnotationBuilder.from_object(annotation_obj)
            annotation.rename_layer('Centre of Tumour', 'Tumour area')
            contours, layer_name = annotation.get_layer_points('Tumour area', contour_format=True)
    except FileNotFoundError as err:
        failure_log.append({
            'file': str(image_path),
            'error': str(err),
            'message': f"No tumour area annotation file: {str(Path(opt.data_dir) / 'data' / opt.area_annotation_dir / (slide_id + '.json'))}"
        })
        print(failure_log[-1]['message'])
        return
    except (KeyError, ValueError) as err:
        failure_log.append({
            'file': str(image_path),
            'error': str(err),
        })
        print(AnnotationBuilder.from_object(annotation_obj).layers)
        return
    # biggest contour is used to select the area to process
    area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3),
                       key=cv2.contourArea)
    # read downsampled region corresponding to tumour area annotation and extract contours
    try:
        slide = WSIReader(image_path, opt)
        mask_slide = WSIReader(mask_path, opt, set_mpp=slide.mpp_x)
    except OpenSlideUnsupportedFormatError:
        return
    assert slide.level_dimensions[0] == mask_slide.level_dimensions[0], \
        "Slides must have same the same resolution at their base level"
    rescale_factor = mask_slide.level_downsamples[mask_slide.read_level]
    x, y, w, h = cv2.boundingRect(area_contour)
    # if base layer is not copied from mask, need to read at half the origin as mask dimensions will be halved
    w_rescaled, h_rescaled = int(w // rescale_factor), int(h // rescale_factor)
    try:
        mask = np.array(mask_slide.read_region((x, y), mask_slide.read_level, (w_rescaled, h_rescaled)))[..., :3]
    except IndexError:
        print(x, y, w, h)
        print(mask)
        print("Empty array")
        return
    # mask = np.zeros((h_rescaled, w_rescaled, 3))
    # nw, nh = w_rescaled // 512, h_rescaled // 512
    # for i in range(nw + 1):  # TODO test
    #     for j in range(nh + 1):
    #         x_len = 512 if i != nw else w_rescaled % 512
    #         y_len = 512 if j != nh else h_rescaled % 512
    #         mask_tile = np.array(
    #             mask_slide.read_region((x + i * round(512 * rescale_factor), y + j * round(512 * rescale_factor)),
    #                                    mask_slide.read_level, (x_len, y_len)))[..., :3]
    #         mask[j * 512:(j + 1) * 512, i * 512:(i + 1) * 512, :] = mask_tile
    converter = MaskConverter()
    print(f"Extracting contours from mask {slide_id}...")
    contours, labels, boxes = converter.mask_to_contour(np.array(mask),
                                                        round(x//rescale_factor) - 1, round(y//rescale_factor) - 1,
                                                        rescale_factor=None)  # don't rescale map inside
    # rescale contours
    if rescale_factor != 1.0:
        print(f"Rescaling contours by {rescale_factor:.2f}")
        contours = [(contour * rescale_factor).astype(np.int32) for contour in contours]
    # contours = [contour + np.array((x, y), dtype=np.int32) for contour in contours]
    layers = tuple(set(labels))
    print("Storing contours into annotation ...")
    annotation = AnnotationBuilder(slide_id, 'extract_contours', layers)
    for contour, label in zip(contours, labels):
        annotation.add_item(label, 'path')
        contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
        annotation.add_segments_to_last_item(contour)
    if len(annotation) == 0:
        warnings.warn(f"No contours were extracted for slide: {slide_id}")
    annotation.shrink_paths(0.1, min_point_density=0.2, min_point_num=20)
    annotation.add_data('experiment', opt.experiment_name)
    annotation.add_data('load_epoch', opt.load_epoch)
    annotation_dir = Path(
        opt.data_dir) / 'data' / 'annotations' / opt.experiment_name
    annotation_dir.mkdir(exist_ok=True, parents=True)
    annotation.dump_to_json(annotation_dir)
    print(f"Annotation saved in {str(annotation_dir)}")


if __name__ == '__main__':
    opt = ProcessOpenSlideOptions().parse()
    failure_log = []

    image_paths = list()
    image_paths += list(path for path in Path(opt.data_dir).glob('*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*.tiff'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.tiff'))
    if opt.shuffle_images:
        import random
        random.shuffle(image_paths)
    pbar = tqdm(total=len(image_paths), desc="slides")

    def update(*a):
        pbar.update()
    extract_annotation_ = partial(extract_annotation, opt=opt, failure_log=failure_log)
    if opt.workers > 0:
        with mp.Pool(opt.workers) as pool:
            promise = pool.map_async(extract_annotation_, image_paths, callback=update)
            result = promise.get()
    else:
        for image_path in image_paths:
            try:
                extract_annotation_(image_path)
            except MemoryError:
                continue



