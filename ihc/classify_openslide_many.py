import warnings
from pathlib import Path
from functools import partial
import socket
import re
from datetime import datetime
import json
from imageio import imread
import numpy as np
import pandas as pd
import torch
from openslide.lowlevel import OpenSlideError, OpenSlideUnsupportedFormatError
from base.options.process_openslide_options import ProcessOpenSlideOptions
from base.models import create_model
from base.inference.wsi_processor import WSIProcessor
from ihc.datasets.ihcpatch_dataset import find_slides_description
from base.utils import debug


def classify_image(images, slide_label, model):
    if isinstance(images, np.ndarray):
        images = [images]
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
    data = {
        'input': images,
        'input_path': '',
        'target': torch.tensor([slide_label]*len(images))
    }
    model.set_input(data)
    model.test()
    outputs = torch.nn.functional.softmax(model.output.detach().cpu(), dim=1).numpy()
    loss = model.loss_bce.detach().cpu().numpy()
    data = {
        'outputs': outputs,
        'loss': loss,
    }
    if hasattr(model, 'variance'):
        variance = model.variance.detach().cpu().numpy().sum(axis=1).tolist()  # reported variance is the sum of variances over classes
        loss_variance = model.loss_variance.detach().cpu().numpy().tolist()
        if model.opt.batch_size == 1:
            loss_variance = [loss_variance]
        data.update(variance=variance, loss_variance=loss_variance)
    if hasattr(model, 'features'):
        data.update(features=model.features.detach().cpu().numpy().reshape(model.opt.batch_size, -1))
    return data


if __name__ == '__main__':
    opt = ProcessOpenSlideOptions().parse(unknown_arg_error=False)
    assert opt.dataset_name == 'ihcpatch'
    print(f"Starting at {str(datetime.now())}")
    print(f"Running on host: '{socket.gethostname()}'")
    opt.no_visdom = True
    opt.phase = 'none'
    slides_data = pd.read_csv(opt.ihc_data_file)
    tiles_dir = Path(opt.data_dir)
    slide_ids, slide_labels, slide_stains = find_slides_description(slides_data, tiles_dir)
    if opt.split_path:
        with open(opt.split_path, 'r') as split_file:
            split = json.load(split_file)
        slide_ids = slide_ids.intersection(set(split['train_slides']).union(set(split['test_slides'])))
    model = create_model(opt)
    model.setup()
    model.eval()
    image_paths = list()
    image_paths += list(path for path in Path(opt.data_dir).glob('*.ndpi') if path.with_suffix('').name in slide_ids)
    image_paths += list(path for path in Path(opt.data_dir).glob('*.svs') if path.with_suffix('').name in slide_ids)
    image_paths += list(path for path in Path(opt.data_dir).glob('*.tiff') if path.with_suffix('').name in slide_ids)
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi') if path.with_suffix('').name in slide_ids)
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs') if path.with_suffix('').name in slide_ids)
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.tiff') if path.with_suffix('').name in slide_ids)
    if opt.debug_slide is not None and len(opt.debug_slide):
        image_paths = [path for path in image_paths if path.with_suffix('').name in opt.debug_slide]
        if len(image_paths) == 0:
            raise ValueError(f"No slides in data dir match debug ids: {opt.debug_slide}")
    print(f"Dirs with slides: {set(path.parent.name for path in image_paths)}")
    if opt.shuffle_images:
        import random
        random.shuffle(image_paths)
    failure_log = []
    try:
        with open(Path(opt.data_dir)/'data'/'thumbnails'/'thumbnails_info.json', 'r') as tiles_info_file:
            thumbnails_info = json.load(tiles_info_file)
    except FileNotFoundError:
        thumbnails_info = None
    classify_image_with_model = partial(classify_image, model=model)
    save_dir = Path(opt.data_dir)/'data'/'classifications'/f'{opt.experiment_name}_{opt.load_epoch}'
    for image_path in image_paths:
        # TODO limit processing to H&E slides by checking additional data file
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        try:
            slide_label = slide_labels[slide_id]
        except KeyError as err:
            print(err)
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Data for {slide_id} is unavailable in {opt.ihc_data_file}"
            })
            continue
        if slide_id not in slide_ids:
            continue
        if slide_id.endswith(('CK5', 'CKAE13', 'CHROMOA', '34B12')):
            continue
        if (save_dir/(slide_id + '.json')).exists() and not opt.overwrite:
            continue
        print(f"Processing slide: {slide_id} (extension: {image_path.suffix})")
        tissue_mask_path = Path(opt.data_dir)/'data'/'masks'/opt.tissue_mask_dirname/f'thumbnail_{image_path.with_suffix("").name}.png'
        try:
            if tissue_mask_path.exists():  # select subset of slide where tissue is present using tissue mask
                with open(tissue_mask_path, 'r') as tissue_mask_file:
                    tissue_mask = imread(tissue_mask_path)
            else:
                tissue_mask = None
                if opt.require_tissue_mask:
                    raise FileNotFoundError(f"No tissue mask available for slide '{slide_id}'")
                else:
                    warnings.warn(f"No tissue mask available for slide '{slide_id}'")
            processor = WSIProcessor(file_name=str(image_path), opt=opt, set_mpp=opt.set_mpp, tissue_mask=tissue_mask,
                                     tissue_mask_info=thumbnails_info[slide_id])
            processor.apply_classification(classify_image_with_model, ['Certain', 'Ambiguous'], slide_label, save_dir)
        except ValueError as err:
            if err.args[0].startswith(('No image locations to process for slide',
                                       'Loaded image is not a binary tissue mask')):
                failure_log.append({
                    'file': str(image_path),
                    'error': str(err),
                    'message': f"Error occurred when applying network to slide {slide_id}"
                })
                continue
            else:
                raise
        except (RuntimeError, KeyError) as err:
            raise
        except (FileNotFoundError, OpenSlideError, OpenSlideUnsupportedFormatError) as err:
            print(err)
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Error occurred when applying network to slide {slide_id}"
            })
            continue
    # save failure log
    logs_dir = Path(opt.data_dir, 'data', 'logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open(logs_dir/f'failures_process_openslide_many_{str(datetime.now())[:10]}_{opt.experiment_name}.json', 'w') as failures_log_file:
        json.dump(failure_log, failures_log_file)
    print("Done!")
