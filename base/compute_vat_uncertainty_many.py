import warnings
from pathlib import Path
from functools import partial
import json
import socket
import re
from datetime import datetime
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from options.compute_vat_uncertainty_options import ComputeVATUncertaintyOptions
from models import create_model
from inference.wsi_processor import WSIProcessor
from annotation.annotation_builder import AnnotationBuilder


def compute_vat(images, model):
    r"""
    Produce soft-maxed network mask that is also visually intelligible.
    Each channel contains the softmax probability of that pixel belonging to that class, mapped to [0-255] RGB values.
    """
    images = np.stack(images, axis=3)
    if images.shape[-2] == 4:
        images = images[:, :, :3, :]
    # scale between 0 and 1
    images = images / 255.0
    # normalised images between -1 and 1
    images = (images - 0.5) / 0.5
    # convert to torch tensor
    images = images.transpose(3, 2, 0, 1)
    images = torch.from_numpy(images.copy()).float()
    data = {'input': images, 'input_path': ['']*images.shape[0]}
    model.set_input(data)
    model.forward()
    vat_sigma = np.array(model.vat_sigma, dtype=np.float32)  # 'logits' as they are not bounded and we want to take softmax on them
    # TODO need to absolute and sum over channel dimension?
    assert images[0].shape[-2:] == vat_sigma.shape[-2:], r"VAT uncertainty tensor must have same spatial dimensions of image"
    vat_sigma = np.tile(np.expand_dims(vat_sigma.transpose(1, 2, 0), 2), (1, 1, 3, 1))
    return [vat_sigma[..., i] for i in range(vat_sigma.shape[-1])]


if __name__ == '__main__':
    opt = ComputeVATUncertaintyOptions().parse()
    print(f"Running on host: '{socket.gethostname()}'")
    opt.no_visdom = True
    model = create_model(opt)
    model.setup()
    model.eval()
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
    failure_log = []
    for image_path in image_paths:
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        print(f"Processing slide: {slide_id} (extension: {image_path.suffix})")
        compute_vat_with_model = partial(compute_vat, model=model)
        #
        uncertainty_map_path = Path(opt.data_dir)/'data'/'uncertainty_maps'/f'{image_path.with_suffix(".tiff").name}'
        if uncertainty_map_path.exists():
            continue
        # read tumour area annotation
        try:
            with open(Path(opt.data_dir) / 'data' / opt.area_annotation_dir /
                      (slide_id + '.json'), 'r') as annotation_file:
                annotation_obj = json.load(annotation_file)
                annotation_obj['slide_id'] = slide_id
                annotation_obj['project_name'] = 'tumour_area'
                annotation_obj['layer_names'] = ['Tumour area']
                contours, layer_name = AnnotationBuilder.from_object(annotation_obj). \
                    get_layer_points('Tumour area', contour_format=True)
        except FileNotFoundError as err:
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"No tumour area annotation file: {str(Path(opt.data_dir) / 'data' / opt.area_annotation_dir / (slide_id + '.json'))}"
            })
            print(failure_log[-1]['message'])
            continue
        # biggest contour is used to select the area to process
        area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3),
                           key=cv2.contourArea)
        if opt.area_contour_rescaling != 1.0:  # rescale annotations that were taken at the non base magnification
            area_contour = (area_contour / opt.area_contour_rescaling).astype(np.int32)
        processor = WSIProcessor(file_name=str(image_path), opt=opt, shift_and_merge=False,
                                 normalize_output=True, filter_location=area_contour)
        processor.apply(compute_vat_with_model, np.float32, Path(opt.data_dir) / 'data' / 'uncertainty_maps')
        # except Exception as err:
        #     failure_log.append({
        #         'file': str(image_path),
        #         'error': str(err),
        #         'message': f"Error occurred when applying network to slide {slide_id}"
        #     })
        #     continue
        # extract contours from processed slide
    # save failure log
    logs_dir = Path(opt.data_dir, 'data', 'logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open(logs_dir/f'failures_compute_vat_uncertainty_many_{str(datetime.now())[:10]}', 'w') as failures_log_file:
        json.dump(failure_log, failures_log_file)

