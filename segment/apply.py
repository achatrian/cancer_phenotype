from pathlib import Path
from datetime import datetime
import socket
import warnings
import re
from itertools import chain
from tqdm import tqdm
import numpy as np
import torch
from imageio import imwrite
from base.options.apply_options import ApplyOptions
from base.models import create_model
from base.datasets import create_dataset, create_dataloader
from annotation.annotation_builder import AnnotationBuilder
from annotation.mask_converter import MaskConverter
from base.utils.utils import segmap2img, bytes2human
from base.utils import debug
torch.manual_seed(0)
np.random.seed(0)


r"""Apply segmentation network to extracted tiles (not full images)"""


if __name__ == '__main__':
    opt = ApplyOptions().parse()
    print(f"Starting at {str(datetime.now())}")
    print(f"Running on host: '{socket.gethostname()}'")
    # hard-code some parameters for test
    opt.no_visdom = True
    opt.sequential_samples = True
    opt.data_dir = Path(opt.data_dir)
    image_paths = []
    for suffix in opt.image_suffix:
        image_paths.extend(chain(opt.data_dir.glob(f'./*.{suffix}'), opt.data_dir.glob(f'*/*.{suffix}')))
    dataset = create_dataset(opt)
    dataset.setup()  # NB swapped in position .make_subset() and .setup()
    dataloader = create_dataloader(dataset)
    model = create_model(opt)
    model.setup(dataset)
    if opt.eval:
        model.eval()
    annotations = {image_path.with_suffix('').name:
                       AnnotationBuilder(image_path.with_suffix('').name, 'segment') for image_path in image_paths}
    converter = MaskConverter()
    masks_dir = Path(opt.data_dir, 'data', 'masks', opt.experiment_name)
    masks_dir.mkdir(exist_ok=True, parents=True)
    print("Begin applying ...")
    for i, data in enumerate(tqdm(dataloader)):
        input_size = data['input'].element_size() * data['input'].nelement()
        if input_size > 100e6:
            print(f"{data['input_paths']} is {bytes2human(input_size)}")
            continue  # some thumbnails can be very large, and might not fit in the gpu
        model.set_input(data)
        model.test()
        output_logits = model.get_current_visuals()['output_map']
        output = torch.nn.functional.softmax(output_logits, dim=1)
        input_images = (model.get_current_visuals()['input_images'].cpu().numpy().transpose(0, 2, 3, 1))
        for j, single_image in enumerate(output):
            if 'x_offset' and 'y_offset' in data:
                x, y = int(data['x_offset'][j]), int(data['y_offset'][j])
            else:
                x, y = 0, 0
            single_image = single_image.detach().cpu().numpy().transpose(1, 2, 0) * 255  # creates image
            single_image = np.around(single_image).astype(np.uint8)  # conversion with uint8 without around
            segmentation_mask = segmap2img(single_image)
            # if __debug__:
            #     debug.show_image(input_images[j])
            #     debug.show_image(segmentation_mask)
            #     print(data['input_path'][j])
            imwrite(masks_dir/Path(data['input_path'][j]).name, segmentation_mask)  # save output segmentation masks
            if opt.extract_contours:
                contours, labels, boxes = converter.mask_to_contour(segmentation_mask, x, y, rescale_factor=None)
                if hasattr(opt, 'zoom_factor') and opt.zoom_factor != 1.0:
                    contours = [(contour / opt.zoom_factor).astype(np.int32) for contour in contours]
                slide_num, core_num = re.match(r'slide(\d+)_core(\d+)', Path(data['input_path'][j]).name).groups()
                #slide_id = Path(data['input_path'][j]).with_suffix('').name
                slide_id = f'slide{slide_num}_core{core_num}'
                annotation = annotations[slide_id]  # name of tile is name of annotation
                for contour, label in zip(contours, labels):
                    if label not in annotation.layers:
                        annotation.add_layer(label)
                    annotation.add_item(label, 'path', points=contour.squeeze().astype(int).tolist())
    if opt.extract_contours:
        for slide_id in annotations:
            annotation = annotations[slide_id]
            if len(annotation) == 0:
                warnings.warn(f"No contours were extracted for slide: {slide_id}")
            annotation.shrink_paths(0.1)
            annotation.add_data('experiment', opt.experiment_name)
            annotation.add_data('load_epoch', opt.load_epoch)
            annotation_dir = Path(opt.data_dir)/'data'/'annotations'/opt.experiment_name
            annotation_dir.mkdir(exist_ok=True, parents=True)
            annotation.dump_to_json(annotation_dir)
            print(f"Annotation saved in {str(annotation_dir)}")
