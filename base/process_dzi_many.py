from functools import partial
from pathlib import Path
import re
from itertools import product
import json
import warnings
import numpy as np
import cv2
import torch
import tqdm
from options.process_dzi_options import ProcessDZIOptions
from models import create_model
from utils.utils import tensor2im
from data.images.dzi_io.tile_generator import TileGenerator
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter


def process_image(image, input_path, model):
    # scale betwesen 0 and 1
    image = image / 255.0
    # normalised images between -1 and 1
    image = (image - 0.5) / 0.5
    # convert to torch tensor
    assert (image.shape[-1] == 3)
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image.copy()).float()
    data = {
        'input': image.unsqueeze(0),
        'input_path': str(input_path)
    }
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    output = tensor2im(visuals['output_map'][0], segmap=True)
    return output


if __name__ == '__main__':
    opt = ProcessDZIOptions().parse()
    opt.display_id = -1   # no visdom display
    model = create_model(opt)
    model.setup()
    model.eval()
    dzi_dir = Path(opt.data_dir)/'data'/'dzi'
    image_paths = list(path for path in Path(opt.data_dir).iterdir()
                       if path.name.endswith('.svs') or path.name.endswith('.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs'))
    failure_log = []
    for image_path in image_paths:
        slide_id = re.sub('\.(ndpi|svs)', '', image_path.name)
        print(f"Processing slide: {slide_id}")
        source_file = re.sub('\.(ndpi|svs)', '.dzi', slide_id)
        source_file = source_file if source_file.endswith('.dzi') else source_file + '.dzi'
        target_file = Path('masks')/('mask_' + Path(source_file).name)
        if (Path(opt.data_dir)/'data'/'dzi'/target_file).exists():
            continue  # skip existing files
        print(f"Source file: {source_file} - Target file: {target_file}")
        source_dir = dzi_dir/(slide_id + '_files')
        process_image_with_model = partial(process_image, input_path=str(source_dir), model=model)
        try:
            dzi = TileGenerator(str(dzi_dir/source_file), target=str(dzi_dir/target_file))
        except FileNotFoundError:
            continue  # move onto next slide
        dzi.clean_target(supress_warning=True)  # cleans the target directory
        dzi.properties['mpp'] = float(dzi.properties['mpp'])
        resize_factor = round(opt.mpp / dzi.properties['mpp'])  # base to target mpp ratio
        read_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) * dzi.properties['mpp'] - opt.mpp)))
        input_size = int(opt.patch_size * resize_factor)
        mask_size = dzi.slide_to_mask((input_size,) * 2)[0]
        xs, ys = list(range(0, dzi.width, input_size)), list(range(0, dzi.height, input_size))
        print("Processing dzi ...")
        for x, y in tqdm.tqdm(product(xs, ys), total=len(xs)*len(ys)):
            x_mask, y_mask = dzi.slide_to_mask((x, y))
            if dzi.masked_percent(x_mask, y_mask, mask_size, mask_size) > 0.3:
                dzi.process_region((x, y), read_level, (opt.patch_size, opt.patch_size), process_image_with_model, border=0)
            else:
                dzi.process_region((x, y), read_level, (opt.patch_size, opt.patch_size), np.zeros_like, border=0)
        dzi.downsample_pyramid(read_level)  # create downsampled levels
        dzi.close()
        print("Processed dzi, now saving annotation ...")
        original_dzi = TileGenerator(str(dzi_dir/source_file))  # NB target is used as source !!
        mask_dzi = TileGenerator(str(dzi_dir/target_file))  # NB target is used as source !!
        original_dzi.properties['mpp'], mask_dzi.properties['mpp'] = (float(original_dzi.properties['mpp']),
                                                                      float(mask_dzi.properties['mpp']))
        converted_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) *
                                                    mask_dzi.properties['mpp'] - opt.mpp)))  # relative to lev
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
        area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3), key=cv2.contourArea)
        if opt.area_contour_rescaling != 1.0:  # rescale annotations that were taken at the non base magnification
            area_contour = (area_contour / opt.area_contour_rescaling).astype(np.int32)
        # read downsampled region corresponding to tumour area annotation and extract contours
        rescale_factor = mask_dzi.properties['mpp'] / original_dzi.properties['mpp']  # to original images
        x, y, w, h = cv2.boundingRect(area_contour)
        # if base layer is not copied from mask, need to read at half the origin as mask dimensions will be halved
        x_read = x if mask_dzi.width == original_dzi.width else int(x / rescale_factor)
        y_read = y if mask_dzi.height == original_dzi.height else int(y / rescale_factor)
        w_read, h_read = int(w / rescale_factor), int(h / rescale_factor)
        mask = mask_dzi.read_region((x_read, y_read), converted_level, (w_read, h_read))
        converter = MaskConverter()
        print("Extracting contours from mask ...")
        contours, labels, boxes = converter.mask_to_contour(mask, x_read, y_read, rescale_factor=None)  # don't rescale map inside
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
        if annotation.is_empty():
            warnings.warn(f"No contours were extracted for slide: {slide_id}")
        annotation.shrink_paths(0.1)
        # add model details to annotation
        annotation.add_data('experiment', opt.experiment_name)
        annotation.add_data('load_epoch', opt.load_epoch)
        annotation_dir = Path(opt.data_dir) / 'data' / 'annotations' / opt.experiment_name
        annotation_dir.mkdir(exist_ok=True, parents=True)
        annotation.dump_to_json(annotation_dir)
        print(f"Annotation saved in {str(annotation_dir)}")
        print("Done !")





