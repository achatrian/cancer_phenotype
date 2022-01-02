import warnings
from pathlib import Path
from functools import partial
import json
import socket
import re
from datetime import datetime
from math import sqrt
import cv2
import numpy as np
import pandas as pd
# from imagecodecs._zlib import ZlibError
# from imagecodecs._deflate import DeflateError
from tifffile import TiffFileError
from openslide.lowlevel import OpenSlideError, OpenSlideUnsupportedFormatError
image_specific_errors = (OpenSlideError, OpenSlideUnsupportedFormatError, TiffFileError) # ZlibError, DeflateError)
from options.process_slides_options import ProcessSlidesOptions
from data.images.wsi_reader import make_wsi_reader, add_reader_args, get_reader_options
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter
from tqdm import tqdm
from base.utils import debug


skip_ending = ('CK5', 'panCK', '34BE12', 'AMACR', 'RACEMASE')


if __name__ == '__main__':
    opt = ProcessSlidesOptions().parse()
    opt.data_dir = Path(opt.data_dir)
    opt.skip_images = set(opt.skip_images)
    opt.gpu_ids = []
    opt.extract_contours = True
    print(f"Starting at {str(datetime.now())}")
    print(f"Running on host: '{socket.gethostname()}'")
    image_paths = []
    save_dir = Path(opt.save_dir) if opt.save_dir is not None else Path(opt.data_dir)/'data'
    for suffix in opt.image_suffix:
        image_paths.extend(opt.data_dir.glob(f'./*.{suffix}'))
        if opt.recursive_search:
            image_paths.extend(opt.data_dir.glob(f'*/*.{suffix}'))
    if opt.debug_slide is not None and len(opt.debug_slide) > 0:
        image_paths = [path for path in image_paths if path.with_suffix('').name in opt.debug_slide]
        if len(image_paths) == 0:
            raise ValueError(f"No slides in data dir match debug ids: {opt.debug_slide}")
    print(f"{len(image_paths)} images to process (extensions: {set(p.suffix for p in image_paths)})")
    if len(image_paths) == 0:
        exit("No paths. Hint: check --image_suffix and --recursive_search")
    all_stains_length = len(image_paths)
    image_paths = [image_path for image_path in image_paths if not image_path.with_suffix('').name.endswith(skip_ending)]
    if opt.cases_list_path is not None:
        cases_list = pd.read_excel(Path(opt.cases_list_path))
        cases_identifiers = tuple(str(id_) for id_ in cases_list['SpecimenIdentifier'])
        image_paths = [image_path for image_path in image_paths if image_path.with_suffix('').name.startswith(cases_identifiers)]
        print(f"Selected {len(image_paths)} paths from {len(cases_identifiers)} cases")
    if opt.slides_list_path is not None:
        slides_list = pd.read_excel(Path(opt.slides_list_path))
        slides_identifiers = set(str(id_) for id_ in slides_list['SlideIdentifier'])
        image_paths = [image_path for image_path in image_paths if image_path.stem in slides_identifiers]
        print(f"Selected {len(image_paths)} slides from {opt.slides_list_path}")
    he_only_length = len(image_paths)
    if he_only_length < all_stains_length:
        print(f"{he_only_length}/{all_stains_length} H&E images")
    if opt.shuffle_images:
        import random
        random.shuffle(image_paths)
    failure_log = []
    try:
        with open(save_dir/'thumbnails'/'thumbnails_info.json', 'r') as tiles_info_file:
            thumbnails_info = json.load(tiles_info_file)
    except FileNotFoundError:
        try:
            with open(save_dir/'masks'/'thumbnails_info.json', 'r') as tiles_info_file:
                thumbnails_info = json.load(tiles_info_file)
        except FileNotFoundError:
            thumbnails_info = None
    annotation_dir = Path(save_dir)/'annotations'/opt.experiment_name
    annotation_dir.mkdir(exist_ok=True, parents=True)
    for image_path in tqdm(image_paths, 'slide'):
        slide_id = re.sub(r'\.(ndpi|svs|tiff|isyntax)', '', image_path.name)
        annotation_path = annotation_dir/(slide_id + '.json')
        if annotation_path.exists():
            continue
        # read tumour area annotation
        try:
            with open(save_dir / opt.area_annotation_dir /
                      (slide_id + '.json'), 'r') as annotation_file:
                annotation_obj = json.load(annotation_file)
                annotation_obj['slide_id'] = slide_id
                annotation_obj['project_name'] = 'tumour_area'
                annotation_obj['layer_names'] = ['Tumour area']
                try:
                    area_contours, layer_name = AnnotationBuilder.from_object(annotation_obj).get_layer_points('Tumour area')
                except ValueError:
                    try:
                        area_contours, layer_name = AnnotationBuilder.from_object(annotation_obj).get_layer_points('Tumour')
                    except ValueError:
                        layer_names = tuple(AnnotationBuilder.from_object(annotation_obj).layers.keys())
                        print(f"No 'Tumour area' or 'Tumour' layers in area annotation (layer names {layer_names})")
                if len(area_contours) == 0:
                    tqdm.write(f"Tumour area annotation for slide '{slide_id}' is empty")
                    continue
        except FileNotFoundError as err:
            tqdm.write(f"No tumour area annotation for slide '{slide_id}'")
            continue
        # read downsampled region corresponding to tumour area annotation and extract contours
        mask_path = save_dir/opt.mask_dirname/image_path.with_suffix('.tiff').name
        try:
            slide = make_wsi_reader(image_path, opt)
        except (FileNotFoundError, IndexError) + image_specific_errors as err:
            print(err)
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Error occurred when applying network to slide {slide_id}"
            })
            continue
        try:
            mask_slide = make_wsi_reader(mask_path, opt, set_mpp=slide.mpp[0])
        except (FileNotFoundError, IndexError) + image_specific_errors as err:
            message = f"No mask available for {image_path.name}"
            print(message)
            failure_log.append({
                'file': str(mask_path),
                'error': str(err),
                'message': message
            })
            continue
        try:
            assert slide.level_dimensions[0] == mask_slide.level_dimensions[0], \
                "Slides must have same the same resolution at their base level"
        except AssertionError:
            print(slide_id, " Slides must have same the same resolution at their base level")
            continue
        rescale_factor = mask_slide.level_downsamples[mask_slide.read_level]
        contours, labels, boxes = [], [], []
        print(f"[{slide_id}] Extracting contours from mask ...")
        for i, area_contour in enumerate(area_contours):
            if area_contour.ndim < 2:
                continue
            x, y, w, h = cv2.boundingRect(area_contour)  # bounding box at base level
            if w*h > opt.max_size_annotation_area:
                area_sampling_ratio = opt.max_size_annotation_area/(w*h)
                w, h = round(sqrt(area_sampling_ratio)*w), round(sqrt(area_sampling_ratio)*h)
                print(f"Cropping annotation area {i} by a factor of {area_sampling_ratio}")
            else:
                area_sampling_ratio = 1.0
            # if base layer is not copied from mask, need to read at half the origin as mask dimensions will be halved
            w_rescaled, h_rescaled = int(w // rescale_factor), int(h // rescale_factor)
            mask = np.zeros((h_rescaled, w_rescaled, 3))
            nw, nh = w_rescaled // 512, h_rescaled // 512
            for i in range(nw + int(w_rescaled % 512 > 0)):  # TODO test
                for j in range(nh + int(h_rescaled % 512 > 0)):
                    x_len = w_rescaled % 512 if i == nw else 512
                    y_len = h_rescaled % 512 if j == nh else 512
                    mask_tile = np.array(
                        mask_slide.read_region(
                            (x + i * int(512 * rescale_factor), y + j * int(512 * rescale_factor)),
                            mask_slide.read_level, (x_len, y_len)))[..., :3]  # TODO FIXME "IndexError: too many indices for array"
                    try:
                        mask[j * 512:(j + 1) * 512, i * 512:(i + 1) * 512, :] = mask_tile
                    except ValueError as err:
                        pass
            converter = MaskConverter()
            x_rescaled, y_rescaled = int(x // rescale_factor), int(y // rescale_factor)
            try:
                contours_, labels_, boxes_ = converter.mask_to_contour(np.array(mask),
                                                                       x_rescaled, y_rescaled,
                                                                       rescale_factor=None)  # don't rescale map inside
            except cv2.error as err:
                failure_log.append({
                    'file': str(image_path),
                    'error': str(err),
                })
                continue
            contours.extend(contours_), labels.extend(labels_), boxes.extend(boxes_)
        # rescale contours
        if rescale_factor != 1.0:
            print(f"[{slide_id}] Rescaling contours by {rescale_factor:.2f}")
            contours = [(contour * rescale_factor).astype(np.int32) for contour in contours]
        layers = tuple(set(labels))
        print(f"[{slide_id}] Storing contours into annotation ...")
        annotation = AnnotationBuilder(slide_id, 'extract_contours', layers)
        for contour, label in zip(contours, labels):
            annotation.add_item(label, 'path')
            contour = contour.squeeze().astype(int).tolist()  # deal with extra dim at pos 1
            annotation.add_segments_to_last_item(contour)
        if len(annotation) == 0:
            warnings.warn(f"[{slide_id}] No contours were extracted for slide: {slide_id}")
        annotation.shrink_paths(0.1)
        annotation.add_data('experiment', opt.experiment_name)
        annotation.add_data('load_epoch', opt.load_epoch)
        annotation.add_data('max_size_annotation_area', opt.max_size_annotation_area)
        annotation.add_data('area_sampling_ratio', area_sampling_ratio)
        annotation.dump_to_json(annotation_dir)
        print(f"Annotation saved in {str(annotation_dir)}")
    # save failure log
    logs_dir = Path(save_dir, 'logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open(logs_dir/f'failures_extract_contours_many_{str(datetime.now())[:10]}', 'w') as failures_log_file:
        json.dump(failure_log, failures_log_file)
    print("Done!")
