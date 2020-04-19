from pathlib import Path
from argparse import ArgumentParser
from random import sample, seed, shuffle
from itertools import product
from imageio import imwrite
from tqdm import tqdm
import numpy as np
import cv2
import torch
from base.options.test_options import TestOptions
from base.models import create_model
from base.utils.utils import tensor2im
from data.images.wsi_reader import WSIReader
from data.contours import read_annotations
from base.utils import debug
seed(42)
np.random.seed(42)  # does this guarantee every model works on the same set of tiles ?


def image_to_tensor(image):
    image = np.array(image)
    if image.shape[-1] == 4:
        image = image[..., :3]
    # scale between 0 and 1
    image = image / 255.0
    # normalised images between -1 and 1
    image = (image - 0.5) / 0.5
    # convert to torch tensor
    assert (image.shape[-1] == 3)
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image.copy()).float().unsqueeze(0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mpp', type=float, default=0.4)
    parser.add_argument('--num_samples', type=int, default=30)
    args, unparsed = parser.parse_known_args()
    opt = TestOptions().parse(unknown_arg_error=False)
    model = create_model(opt)
    model.setup()
    if opt.eval:
        model.eval()
    layer_name = 'epithelium'
    save_dir = Path(opt.checkpoints_dir, opt.experiment_name, 'sample_results')
    save_dir.mkdir(exist_ok=True, parents=True)
    image_paths = list()
    image_paths += list(path for path in Path(opt.data_dir).glob('*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*.tiff'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.tiff'))
    image_paths = sample(image_paths, args.num_samples)
    slides = [WSIReader(image_path, {'mpp': args.mpp}) for image_path in image_paths]
    contour_struct = read_annotations(opt.data_dir,
                                      tuple(image_path.with_suffix('').name for image_path in image_paths),
                                      annotation_dirname='tumour_area_annotations')
    for slide, image_path in tqdm(zip(slides, image_paths)):
        try:
            contours = contour_struct[image_path.with_suffix('').name]['Tumour area']
        except KeyError:
            continue
        area_contour = max((contour for contour in contours if contour.shape[0] > 1 and contour.ndim == 3),
                           key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(area_contour)
        xs = list(range(0, slide.level_dimensions[0][0], min(opt.patch_size, w/2)))
        ys = list(range(0, slide.level_dimensions[0][1], min(opt.patch_size, w/2)))
        points = list(product(xs, ys))
        shuffle(points)
        for point in points:
            if cv2.pointPolygonTest(area_contour, tuple(point), False) >= 0.0 and \
                    cv2.pointPolygonTest(area_contour, (point[0] + opt.patch_size, point[1]), False) >= 0.0 and \
                    cv2.pointPolygonTest(area_contour, (point[0] + opt.patch_size, point[1] + opt.patch_size), False) >= 0.0 and \
                    cv2.pointPolygonTest(area_contour, (point[0], point[1] + opt.patch_size), False) >= 0.0:
                break
        else:
            raise ValueError("There are no points in the tumour area")
        tile = slide.read_region(point, slide.read_level, (opt.patch_size,)*2)
        model.set_input({'input': image_to_tensor(tile), 'input_path': str(image_path)})
        model.test()
        visuals = model.get_current_visuals()
        mask = tensor2im(visuals['output_map'][0], segmap=True)
        imwrite(save_dir/(image_path.with_suffix('').name + '.png'), tile)
        imwrite(save_dir/(image_path.with_suffix('').name + '_mask.png'), mask)
    print("Done!")
