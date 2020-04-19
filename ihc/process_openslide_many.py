from pathlib import Path
from functools import partial
import socket
import re
from datetime import datetime
import numpy as np
import torch
from openslide.lowlevel import OpenSlideError
from base.options.process_openslide_options import ProcessOpenSlideOptions
from base.models import create_model
from base.inference.wsi_processor__ import WSIProcessor
from annotation.annotation_builder import AnnotationBuilder


def classify_image(images, model):
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
    data = {'input': images, 'input_path': ''}
    model.set_input(data)
    model.test()
    outputs = model.output.cpu().numpy()
    losses = model.loss_bce.cpu().numpy()
    return outputs, losses


if __name__ == '__main__':
    opt = ProcessOpenSlideOptions().parse()
    print(f"Starting at {str(datetime.now())}")
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
    classify_image_with_model = partial(classify_image, model=model)
    save_dir = Path(opt.data_dir)/'data'/'classifications'/f'{opt.model}_{opt.load_epoch}'
    for image_path in image_paths:
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        print(f"Processing slide: {slide_id} (extension: {image_path.suffix})")
        try:
            processor = WSIProcessor(file_name=str(image_path), opt=opt)
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
        processor.apply_classification(classify_image_with_model, ['Certain', 'Ambiguous'], save_dir)

