from pathlib import Path
from functools import partial
import socket
import re
from datetime import datetime
import json
import numpy as np
import torch
from openslide.lowlevel import OpenSlideError
from base.options.process_openslide_options import ProcessOpenSlideOptions
from base.models import create_model
from base.inference.wsi_processor import WSIProcessor
from ihc.datasets.ihcpatch_dataset import IHCPatchDataset


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
    return outputs, loss


if __name__ == '__main__':
    opt = ProcessOpenSlideOptions().parse(unknown_arg_error=False)
    assert opt.dataset_name == 'ihcpatch'
    print(f"Starting at {str(datetime.now())}")
    print(f"Running on host: '{socket.gethostname()}'")
    opt.no_visdom = True
    opt.phase = 'none'
    dataset = IHCPatchDataset(opt)
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
    save_dir = Path(opt.data_dir)/'data'/'classifications'/f'{opt.experiment_name}_{opt.load_epoch}'
    for image_path in image_paths:
        # TODO limit processing to H&E slides by checking additional data file
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        if (save_dir/(slide_id + '.json')).exists():
            continue
        print(f"Processing slide: {slide_id} (extension: {image_path.suffix})")
        try:
            processor = WSIProcessor(file_name=str(image_path), opt=opt, set_mpp=opt.set_mpp)
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
        try:
            slide_label = dataset.slide_labels[slide_id]
        except KeyError as err:
            failure_log.append({
                'file': str(image_path),
                'error': str(err),
                'message': f"Directory was not created during tiles export {slide_id}"
            })
            continue
        processor.apply_classification(classify_image_with_model, ['Certain', 'Ambiguous'], slide_label, save_dir)
    # save failure log
    logs_dir = Path(opt.data_dir, 'data', 'logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open(logs_dir/f'failures_process_openslide_many_{str(datetime.now())[:10]}', 'w') as failures_log_file:
        json.dump(failure_log, failures_log_file)
    print("Done!")

