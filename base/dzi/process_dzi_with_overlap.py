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
from utils.utils import segmap2img
from data.images.dzi_io.tile_generator import TileGenerator
from data.images.dzi_io import DZIIO, DZISequential
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter


def process_image(image, input_path, model):
    r"""
    Produce soft-maxed network mask that is also visually intelligible.
    Each channel contains the softmax probability of that pixel belonging to that class, mapped to [0-255] RGB values.
    """
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
    output_logits = model.get_current_visuals()['output_map'][0]  # 'logits' as they are not bounded and we want to take softmax on them
    if output_logits.shape[0] > 3:
        raise ValueError(f"Since segmentation has {output_logits.shape[0]} > 3 classes, unintelligible dzi images would be produced from combination.")
    output = torch.nn.functional.softmax(output_logits, dim=0)
    output = output.detach().cpu().numpy().transpose(1, 2, 0) * 255  # creates image
    output = np.around(output).astype(np.uint8)  # conversion with uint8 without arouind
    return output


def merge_tiles(mask0, mask1, overlap=0.5):
    x, y = np.meshgrid(np.arange(mask0.shape[1]), np.arange(mask0.shape[0]))

    def w(x, y):
        r"""Line increase from 0 to 1 at overlap end, plateau at 1 in center,
        and decay from overlap start to end of image"""
        lx, ly, r = (mask0.shape[1]-1), (mask0.shape[0]-1), overlap
        if 0 <= x < lx*r:
            wx = x/(lx*r)
        elif lx*r <= x < lx*(1-r):
            wx = 1.0
        elif lx*(1-r) <= x <= lx:
            wx = -x/(lx*r) + 1/r
        else:
            raise ValueError(f"x must be in range [{0}, {lx}]")
        if 0 <= y < ly*r:
            wy = y/(ly*r)
        elif ly*r <= y < ly*(1-r):
            wy = 1.0
        elif ly*(1-r) <= y <= ly:
            wy = -y/(ly*r) + 1/r
        else:
            raise ValueError(f"y must be in range [{0}, {ly}]")
        return wx*wy
    w = np.vectorize(w)
    weights0 = np.tile(w(x, y)[..., np.newaxis], (1, 1, 3))
    weights1 = np.ones_like(mask0) - weights0  # complementary, so w0(x, y) + w1(x, y) = 1
    mean_mask = (weights0*mask0/255.0 + weights1*mask1/255.0)/2
    image = segmap2img(mean_mask)
    return np.tile(image[..., np.newaxis], (1, 1, 3))


if __name__ == '__main__':
    options = ProcessDZIOptions()
    options.parser.add_argument('--tile_overlap', type=float, default=0.5, help="Determines how big the second dzi offset will be, so that overlap of tiles is as desired")
    opt = options.parse()
    opt.display_id = -1   # no visdom display
    model = create_model(opt)
    model.setup()
    model.eval()
    dzi_dir = Path(opt.data_dir)/'data'/'dzi'
    source_file = re.sub('\.(ndpi|svs)', '.dzi', opt.slide_id)
    source_file = source_file if source_file.endswith('.dzi') else source_file + '.dzi'
    # first dzi -- start from origin corner in original dzi
    target_file = Path('prob_masks')/('prob_mask_' + Path(source_file).name)
    print(f"Source file: {source_file} - Target file: {target_file}")
    source_dir = dzi_dir/(opt.slide_id + '_files')
    process_image_with_model = partial(process_image, input_path=str(source_dir), model=model)
    dzi = TileGenerator(str(dzi_dir/source_file), target=str(dzi_dir/target_file))
    dzi.clean_target(supress_warning=True)  # cleans the target directory
    dzi.properties['mpp'] = float(dzi.properties['mpp'])
    resize_factor = round(opt.mpp / dzi.properties['mpp'])  # base to target mpp ratio
    read_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) * dzi.properties['mpp'] - opt.mpp)))
    input_size = int(opt.patch_size * resize_factor)
    mask_size = dzi.slide_to_mask((input_size,) * 2)[0]
    xs, ys = list(range(0, dzi.width, input_size)), list(range(0, dzi.height, input_size))
    print("Processing origin-aligned dzi ...")
    for x, y in tqdm.tqdm(product(xs, ys), total=len(xs)*len(ys)):
        x_mask, y_mask = dzi.slide_to_mask((x, y))
        if dzi.masked_percent(x_mask, y_mask, mask_size, mask_size) > opt.tissue_content_threshold:
            dzi.process_region((x, y), read_level, (opt.patch_size, opt.patch_size), process_image_with_model, border=0)
        else:
            dzi.process_region((x, y), read_level, (opt.patch_size, opt.patch_size), np.zeros_like, border=0)
    dzi.downsample_pyramid(read_level)  # create downsampled levels
    dzi.close()
    # second dzi -- start from origin corner in original dzi
    target_file = Path(f'prob_masks_shifted_{opt.tile_overlap}')/(f'prob_masks_shifted_{opt.tile_overlap}_' + Path(source_file).name)
    dzi = TileGenerator(str(dzi_dir / source_file), target=str(dzi_dir / target_file))
    dzi.clean_target(supress_warning=True)  # cleans the target directory
    dzi.properties['mpp'] = float(dzi.properties['mpp'])
    resize_factor = round(opt.mpp / dzi.properties['mpp'])  # base to target mpp ratio
    read_level = int(np.argmin(np.absolute(np.power(2, np.arange(0, 6)) * dzi.properties['mpp'] - opt.mpp)))
    input_size = int(opt.patch_size * resize_factor)
    mask_size = dzi.slide_to_mask((input_size,) * 2)[0]
    xs, ys = list(range(0, dzi.width, input_size)), list(range(0, dzi.height, input_size))
    # below: add the first position to coordinate lists, so that sequential can overlap images
    xs, ys = (xs[0],) + tuple(int(x + opt.tile_overlap * opt.patch_size) for x in xs[:-1]), \
             (ys[0],) + tuple(int(y + opt.tile_overlap * opt.patch_size) for y in ys[:-1])  # skip last entry as it's outside of image
    print(f"Processing shifted dzi - overlap of {opt.tile_overlap}%")
    for x, y in tqdm.tqdm(product(xs, ys), total=len(xs) * len(ys)):
        x_mask, y_mask = dzi.slide_to_mask((x, y))
        if dzi.masked_percent(x_mask, y_mask, mask_size, mask_size) > 0.3:
            dzi.process_region((x, y), read_level, (opt.patch_size, opt.patch_size), process_image_with_model, border=0)
        else:
            dzi.process_region((x, y), read_level, (opt.patch_size, opt.patch_size), np.zeros_like, border=0)
    dzi.downsample_pyramid(read_level)  # create downsampled levels
    dzi.close()
    print("Combining shifted dzi images ...")
    target_file = Path('overlap_masks')/('mask_' + Path(source_file).name)
    prob_dzi = DZIIO(
        src=str(dzi_dir/'prob_masks'/('prob_mask_' + Path(source_file).name)),
        target=str(dzi_dir/target_file)
    )  # TODO target dir for new slide must be specified here, but Ka Ho will change this in next update: target dir will be passed directly to Sequential
    prob_dzi_shifted = DZIIO(str(dzi_dir/f'prob_masks_shifted_{opt.tile_overlap}'/(f'prob_masks_shifted_{opt.tile_overlap}_'
                                                              + Path(source_file).name)))
    dzi_seq = DZISequential((prob_dzi, prob_dzi_shifted), merge_tiles)
    dzi_seq.evaluate()
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
        annotation_obj['slide_id'] = opt.slide_id
        annotation_obj['project_name'] = 'tumour_area'
        annotation_obj['layer_names'] = ['Tumour area']
        contours, layer_name = AnnotationBuilder.from_object(annotation_obj). \
            get_layer_points('Tumour area', contour_format=True)
    # biggest contour is used to select the area to process
    area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3),
                       key=cv2.contourArea)
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
