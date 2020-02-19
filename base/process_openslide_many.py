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
from openslide.lowlevel import OpenSlideError
from options.process_openslide_options import ProcessOpenSlideOptions
from models import create_model
from inference.wsi_processor import WSIProcessor
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter


def process_image(image, input_path, model):
    r"""
    Produce soft-maxed network mask that is also visually intelligible.
    Each channel contains the softmax probability of that pixel belonging to that class, mapped to [0-255] RGB values.
    """
    if image.shape[-1] == 4:
        image = image[..., :3]
    assert image.shape[-1] == 3
    # scale between 0 and 1
    image = image / 255.0
    # normalised images between -1 and 1
    image = (image - 0.5) / 0.5
    # convert to torch tensor
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image.copy()).float()
    data = {'input': image.unsqueeze(0), 'input_path': str(input_path)}
    model.set_input(data)
    model.test()
    output_logits = model.get_current_visuals()['output_map'][0]  # 'logits' as they are not bounded and we want to take softmax on them
    if output_logits.shape[0] > 3:
        raise ValueError(f"Since segmentation has {output_logits.shape[0]} > 3 classes, unintelligible dzi images would be produced from combination.")
    output = torch.nn.functional.softmax(output_logits, dim=0)
    output = output.detach().cpu().numpy().transpose(1, 2, 0) * 255  # creates image
    output = np.around(output).astype(np.uint8)  # conversion with uint8 without around
    return output


if __name__ == '__main__':
    opt = ProcessOpenSlideOptions().parse()
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
        process_image_with_model = partial(process_image, input_path=image_path, model=model)
        try:
            processor = WSIProcessor(file_name=str(image_path), opt=opt)
            processor.apply(process_image_with_model, np.uint8, Path(opt.data_dir)/'data'/'masks')
        except (RuntimeError, ValueError, KeyError) as err:
            raise
        except (FileNotFoundError, OpenSlideError) as err:
            print(err)
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Error occurred when applying network to slide {slide_id}"
            })
            continue
        # extract contours from processed slide
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
        # read downsampled region corresponding to tumour area annotation and extract contours
        mask_slide = WSIProcessor(Path(opt.data_dir)/'data'/'masks'/image_path.with_suffix('.tiff').name, opt)
        rescale_factor = processor.level_downsamples[processor.read_level]  # to original images
        x, y, w, h = cv2.boundingRect(area_contour)
        # if base layer is not copied from mask, need to read at half the origin as mask dimensions will be halved
        mask = mask_slide.read_region((x, y), mask_slide.read_level, (w//rescale_factor, h//rescale_factor))
        converter = MaskConverter()
        print("Extracting contours from mask ...")
        contours, labels, boxes = converter.mask_to_contour(mask, x, y, rescale_factor=None)  # don't rescale map inside
        # rescale contours
        if rescale_factor != 1.0:
            print(f"Rescaling contours by {rescale_factor:.2f}")
            contours = [(contour * rescale_factor).astype(np.int32) for contour in contours]
        layers = tuple(set(labels))
        print("Storing contours into annotation ...")
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
        annotation_dir = Path(opt.data_dir) / 'data' / 'annotations_TEST' / opt.experiment_name  # TODO change to 'annotations' if this works
        annotation_dir.mkdir(exist_ok=True, parents=True)
        annotation.dump_to_json(annotation_dir)
        print(f"Annotation saved in {str(annotation_dir)}")
    # save failure log
    logs_dir = Path(opt.data_dir, 'data', 'logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open(logs_dir/f'failures_process_openslide_many_{str(datetime.now())[:10]}', 'w') as failures_log_file:
        json.dump(failure_log, failures_log_file)

