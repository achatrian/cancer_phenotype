from typing import Union
from pathlib import Path
from numbers import Number
import json
import cv2
import numpy as np
import imageio
from tqdm import tqdm
from data.images.wsi_reader import WSIReader
from data import read_annotations, get_contour_image
from data.instance_masker import InstanceMasker


# from data.images.dzi_io.dzi_io import DZI_IO


class InstanceTileExporter:
    r"""Extract tiles centered around images component instances.
    If an instance is larger than the given tile size, multiple tiles per instance are extracted"""
    def __init__(self, data_dir, slide_id, tile_size=1024, mpp=0.2, label_values=(('epithelium', 200), ('lumen', 250))):
        self.data_dir = Path(data_dir)
        self.slide_id = slide_id
        self.tile_size = tile_size
        self.mpp = mpp
        self.label_values = dict(label_values)
        annotations_path = Path(self.data_dir)/'data'/'annotations'
        try:
            self.slide_path = next(path for path in self.data_dir.iterdir() if slide_id in path.name
                                   and path.name.endswith(('.svs', '.ndpi', '.tiff')))
        except StopIteration:
            raise ValueError(f"No image file matching id: {slide_id}")
        try:
            self.annotation_path = next(path for path in annotations_path.iterdir() if slide_id in path.name)
        except StopIteration:
            raise ValueError(f"No annotation matching slide id: {slide_id}")
        slide_opt = WSIReader.get_reader_options(False, False, args=(f'--mpp={mpp}',))
        self.slide = WSIReader(self.slide_path, slide_opt)
        contour_struct = read_annotations(self.data_dir, slide_ids=(self.slide_id,))
        self.contour_lib = contour_struct[self.slide_id]
        self.tile_size = tile_size
        self.center_crop = CenterCrop(self.tile_size)

    def export_tiles(self, layer: Union[str, int], save_dir: Union[str, Path]):
        r"""
        :param layer: layer name of contours to extract images for
        :param save_dir: save directory for images
        :return:s
        """
        save_dir = Path(save_dir)/layer
        slide_dir = save_dir/self.slide_id
        save_dir.mkdir(exist_ok=True, parents=True)
        slide_dir.mkdir(exist_ok=True)
        masker = InstanceMasker(self.contour_lib,
                                  layer,  # new instance with selected outer layer
                                  label_values=self.label_values)
        for i, contour in enumerate(tqdm(masker.outer_contours)):
            image = get_contour_image(contour, self.slide, min_size=(self.tile_size,)*2 if self.tile_size else ())
            mask, components = masker.get_shaped_mask(i, shape=image.shape)
            mask = cv2.dilate(mask, np.ones((3, 3)))  # pre-dilate to remove jagged boundary from low-res contour extraction
            x, y, w, h = cv2.boundingRect(components['parent_contour'])
            assert image.shape[0:2] == mask.shape[0:2], "Image and mask must be of the same size"
            images, masks = self.fit_to_size(image, mask)
            for j, (image, mask) in enumerate(zip(images, masks)):
                name = f'{layer}_{int(x)}_{int(y)}_{int(w)}_{int(h)}' + '_' + ('' if len(images) == 1 else str(j))
                imageio.imwrite(slide_dir/(name + '_image.png'), image.astype(np.uint8))
                if mask.shape[2] == 3:
                    mask = mask[..., 0]  # cv2.dilate is adding channels?
                imageio.imwrite(slide_dir/(name + '_mask.png'), mask.astype(np.uint8))
        (save_dir/'logs').mkdir(exist_ok=True)
        with open(save_dir/'logs'/f'{self.slide_id}_tiles.json', 'w') as tiles_file:
            json.dump({
                'slide_id': self.slide_id,
                'mpp': self.mpp,
                'tile_size': self.tile_size,
                'layer': layer
            }, tiles_file)

    def fit_to_size(self, image, mask, multitile_threshold=2, mask_fill_threshold=0.3):
        r"""
        Divides up image and corresponding mask into tiles of equal size
        :param image:
        :param mask:
        :param multitile_threshold: smallest
        :param mask_fill_threshold:
        :return:
        """
        too_narrow = image.shape[1] < self.tile_size
        too_short = image.shape[0] < self.tile_size
        if too_narrow or too_short:
            # pad if needed
            delta_w = self.tile_size - image.shape[1] if too_narrow else 0
            delta_h = self.tile_size - image.shape[0] if too_short else 0
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            images = [cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT)]
            masks = [cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_REFLECT)]
        elif image.shape[0] > self.tile_size or image.shape[1] > self.tile_size:
            if image.shape[0] > self.tile_size * multitile_threshold or \
                    image.shape[1] > self.tile_size * multitile_threshold:
                images, masks = [], []
                for i in range(0, image.shape[0], self.tile_size):
                    for j in range(0, image.shape[1], self.tile_size):
                        r = i if i + self.tile_size < image.shape[0] else image.shape[0] - self.tile_size
                        c = j if j + self.tile_size < image.shape[1] else image.shape[1] - self.tile_size
                        submask = mask[r:r+self.tile_size, c:c+self.tile_size]
                        if (submask > 0).sum()/submask.size < mask_fill_threshold:
                            continue
                        images.append(image[r:r+self.tile_size, c:c+self.tile_size])
                        masks.append(submask)
            else:
                images = [self.center_crop(image)]
                masks = [self.center_crop(mask)]
        else:
            images = [image]
            masks = [mask]
        assert all(image.shape[:2] == (self.tile_size, self.tile_size) for image in images), "images must be of specified shape"
        return images, masks


class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img[y1:y1+th, x1:x1+tw, ...]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('slide_id', type=str)
    parser.add_argument('--mpp', type=float, default=1.0)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--outer_label', type=str, default='epithelium', help="Layer whose instances are returned, one per tiles")
    parser.add_argument('--label_values', type=json.loads, default='[["epithelium", 200], ["lumen", 250]]',
                        help='!!! NB: this would be "[[\"epithelium\", 200], [\"lumen\", 250]]" if passed externally')
    args = parser.parse_args()
    tiler = InstanceTileExporter(args.data_dir, args.slide_id, args.tile_size, args.mpp, args.label_values)
    tiler.export_tiles(args.outer_label, args.data_dir/'data'/'tiles')
