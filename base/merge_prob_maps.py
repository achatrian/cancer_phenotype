import warnings
from pathlib import Path
from functools import partial
import json
import socket
import re
from datetime import datetime
from sys import version_info
from tifffile import __version__ as tifffile_version
from tifffile.tifffile import TiffFileError
import cv2
import torch
import numpy as np
import pandas as pd
from imageio import imread
try:
    from imagecodecs._zlib import ZlibError
    from imagecodecs._deflate import DeflateError
    from tifffile import TiffFileError
    from openslide.lowlevel import OpenSlideError, OpenSlideUnsupportedFormatError
    image_specific_error = True
except ImportError:
    image_specific_error = False
from skimage.morphology import remove_small_objects, remove_small_holes
from options.process_slides_options import ProcessSlidesOptions
from models import create_model
from data.images.wsi_reader import make_wsi_reader, add_reader_args, get_reader_options
from inference.wsi_processor import WSIProcessor
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter
from tqdm import tqdm
from base.utils import debug


skip_ending = ('CK5', 'panCK', '34BE12', 'AMACR', 'RACEMASE')


if __name__ == '__main__':
    opt = ProcessSlidesOptions().parse()
    opt.data_dir = Path(opt.data_dir)
    opt.skip_images = set(opt.skip_images)
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
            raise ValueError(f"No slides in data dir match debug ids: {opt.slide_id}")
    print(f"{len(image_paths)} images to process (extensions: {set(p.suffix for p in image_paths)})")
    if len(image_paths) == 0:
        print("Hint: check --image_suffix and --recursive_search")
    all_stains_length = len(image_paths)
    image_paths = [image_path for image_path in image_paths if not image_path.with_suffix('').name.endswith(skip_ending)]
    he_only_length = len(image_paths)
    if he_only_length < all_stains_length:
        print(f"{he_only_length}/{all_stains_length} H&E images")
    if opt.cases_list_path is not None:
        cases_list = pd.read_excel(Path(opt.cases_list_path))
        cases_identifiers = tuple(str(id_) for id_ in cases_list['SpecimenIdentifier'])
        image_paths = [image_path for image_path in image_paths if image_path.with_suffix('').name.startswith(cases_identifiers)]
        print(f"Selected {len(image_paths)} paths from {len(cases_identifiers)} cases from {opt.cases_list_path}")
    if opt.slides_list_path is not None:
        slides_list = pd.read_excel(Path(opt.slides_list_path))
        slides_identifiers = set(str(id_) for id_ in slides_list['SlideIdentifier'])
        image_paths = [image_path for image_path in image_paths if image_path.stem in slides_identifiers]
        print(f"Selected {len(image_paths)} slides from {opt.slides_list_path}")
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
    annotation_dir = Path(opt.save_dir)/'annotations'/opt.experiment_name
    annotation_dir.mkdir(exist_ok=True, parents=True)
    if image_specific_error:
        image_specific_errors = (OpenSlideError, OpenSlideUnsupportedFormatError, ZlibError, DeflateError, TiffFileError)
    else:
        image_specific_errors = ()
    for image_path in tqdm(image_paths, 'slide'):
        if image_path.with_suffix('').name in opt.skip_images:
            continue
        mask_path = save_dir/opt.mask_dirname/image_path.with_suffix('.tiff').name
        annotation_path = annotation_dir/image_path.with_suffix('.json').name
        # segmentation mask - if it already exists continue to next image
        if mask_path.exists() and (not opt.extract_contours or annotation_path.exists()):
            continue
        slide_id = re.sub(r'\.(ndpi|svs|tiff|isyntax)', '', image_path.name)
        print(f"Processing slide: {slide_id} (extension: {image_path.suffix})")
        # check whether tissue mask exists for slide and if so load it
        tissue_mask_path = save_dir/'masks'/opt.tissue_mask_dirname/f'mask_{image_path.with_suffix("").name}.png'
        if not tissue_mask_path.exists():
            tissue_mask_path = save_dir/'masks'/opt.tissue_mask_dirname/f'{image_path.with_suffix("").name}.png'
        success = False
        try:
            if tissue_mask_path.exists():  # select subset of slide where tissue is present using tissue mask
                with open(tissue_mask_path, 'r') as tissue_mask_file:
                    tissue_mask = imread(tissue_mask_path)  # tissue masks were saved with a bias of 100
            else:
                tissue_mask = None
                if opt.require_tissue_mask:
                    raise FileNotFoundError(f"No tissue mask available for slide '{slide_id}'")
                else:
                    # if no tissue mask is present, do not process the slide
                    warnings.warn(f"No tissue mask available for slide '{slide_id}'")
                    continue
            processor = WSIProcessor(file_name=str(image_path), opt=opt, set_mpp=opt.set_mpp, tissue_mask=tissue_mask,
                                     tissue_mask_info=thumbnails_info[slide_id])
            def a():
                pass
            success = processor.apply(a, np.uint8, save_dir/opt.mask_dirname)
        except ValueError as err:
            if err.args[0].startswith('No image locations to process for slide'):
                failure_log.append({
                    'file': str(image_path),
                    'error': str(err),
                    'message': f"Error occurred when applying network to slide {slide_id}"
                })
                continue
            else:
                raise
        except (RuntimeError) as err:
            message = f"Possible PixelEngine error for '{slide_id}' - check for other runtime errors"
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': message
            })
            print(message)
        except KeyError as err:
            raise
        except (FileNotFoundError, IndexError) + image_specific_errors as err:
            print(err)
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Error occurred when applying network to slide {slide_id}"
            })
            continue
        except TiffFileError as err:
            print(err)
            failure_log.append({
                '   file': str(image_path),
                'error': str(err),
                'message': f"TiffError when opening slide '{slide_id}'"
            })
        if not success:  # if slides don't contain tissue tiles
            continue
        # extract contours from processed slide
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
                    area_contours, layer_name = AnnotationBuilder.from_object(annotation_obj).get_layer_points('Tumour')
        except FileNotFoundError as err:
            if tissue_mask is not None:
                if int(cv2.__version__.split('.')[0]) == 3:
                    _, area_contours, hierarchy = cv2.findContours(tissue_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                else:
                    area_contours, hierarchy = cv2.findContours(tissue_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if len(area_contours) > 20:
                    # only keep contours above 95% quantile
                    areas = [cv2.contourArea(contour) for contour in area_contours]
                    area_contours = [area_contour for area_contour, area in zip(area_contours, areas)
                                     if area > np.quantile(areas, 0.95)]
                # if __debug__:
                #     tissue_mask_copy = tissue_mask.copy().astype(np.int32)
                #     cv2.drawContours(tissue_mask_copy, area_contours, -1, (100, 0, 0), 3)
                #     debug.show_image(tissue_mask_copy)
                # rescale contours to base level size
                tissue_mask_downsample = thumbnails_info[slide_id]['mpp'][0]/thumbnails_info[slide_id]['read_mpp']
                area_contours = [(area_contour*tissue_mask_downsample).round().astype(np.int32)
                                 for area_contour in area_contours]
            else:
                failure_log.append({
                    'file': str(image_path),
                    'error': str(err),
                    'message': f"No tumour area annotation file: {str(Path(opt.data_dir) / 'data' / opt.area_annotation_dir / (slide_id + '.json'))}"
                })
                print(failure_log[-1]['message'])
                continue
        except (KeyError, ValueError) as err:
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
            })
            continue
    # save failure log
    logs_dir = Path(save_dir, 'logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open(logs_dir/f'failures_process_slides_many_{str(datetime.now())[:10]}', 'w') as failures_log_file:
        json.dump(failure_log, failures_log_file)
    print("Done!")
