from typing import Union
from pathlib import Path
from numbers import Number
import warnings
import cv2
import numpy as np
from skimage import color
import imageio
from base.data.wsi_reader import WSIReader
from base.utils.annotation_builder import AnnotationBuilder
from dzi_io.dzi_io import DZI_IO


class TileExporter:
    # TODO test
    def __init__(self, annotation: AnnotationBuilder, reader: Union[WSIReader, DZI_IO], tile_size=None):
        self.annotation = annotation
        self.reader = reader
        self.tile_size = tile_size
        self.center_crop = CenterCrop(self.tile_size)

    @staticmethod
    def get_contour_image(contour: np.array, reader: Union[WSIReader, DZI_IO]):
        r"""
        Extract image from slide corresponding to region covered by contour
        :param contour: area of interest
        :param reader: object implementing .read_region to extract the desired image
        """
        x, y, w, h = cv2.boundingRect(contour)
        # level below: annotation coordinates should refer to lowest level
        image = np.array(reader.read_region((x, y), level=0, size=(w, h)))
        if image.shape[-2] == 4:
            image = color.rgba2rgb(image)  # RGBA to RGB TODO this failed feature.is_image() test
        return image

    @staticmethod
    def contour_to_mask(contour: np.ndarray, value=250, shape=(), mask=None, mask_origin=None,
                        fit_to_size='contour'):
        r"""Convert a contour to the corresponding max - mask is
        :param contour:
        :param value:
        :param shape: shape of output mask. If not given, mask is as large as contour
        :param pad: amount
        :param mask: mask onto which to paint the new contour
        :param mask_origin: position of mask in slide coordinates
        :param fit_to_size: whether to crop contour to mask if mask is too small or mask to contour if contour is too big
        :return:
        """
        assert type(contour) is np.ndarray and contour.size > 0, "Numpy array expected for contour"
        assert fit_to_size in ('mask', 'contour'), "Invalid value for fit_to_size: " + str(fit_to_size)
        contour = contour.squeeze()
        if isinstance(mask, np.ndarray) and mask_origin:
            assert len(mask_origin) == 2, "Must be x and y coordinate of mask offset"
            contour = contour - np.array(mask_origin)
        else:
            contour = contour - contour.min(0)  # remove slide offset (don't modify internal reference)
        # below: dimensions of contour to image coords (+1's are to match bounding box dims from cv2.boundingRect)
        contour_dims = (contour[:, 1].max() + 1, contour[:, 0].max() + 1)  # xy to row-columns (rc) coordinates
        shape = mask.shape if not shape and isinstance(mask, np.ndarray) else contour_dims
        if mask is None:
            mask = np.zeros(shape)
        y_diff, x_diff = contour_dims[0] - shape[0], contour_dims[1] - shape[1]
        if fit_to_size == 'contour':
            cut_points = []  # find all the indices of points that would fall outside of mask
            if y_diff > 0:
                cut_points.extend(np.where(contour[:, 1].squeeze() > shape[0])[0])  # y to row
            if x_diff > 0:
                cut_points.extend(np.where(contour[:, 0].squeeze() > shape[1])[0])  # x to column
            points_to_keep = sorted(set(range(contour.shape[0])) - set(cut_points))
            if len(points_to_keep) == 0:
                raise ValueError(f"Contour and mask do not overlap (contour origin {contour.min(0)}, mask shape {shape}, mask origin {mask_origin})")
            contour = contour[points_to_keep, :]
            contour_dims = (contour[:, 1].max() + 1, contour[:, 0].max() + 1)  # xy to rc coordinates
        elif fit_to_size == 'mask':
            pad_width = ((0, y_diff if y_diff > 0 else 0), (0, x_diff if x_diff > 0 else 0), (0, 0))
            mask = np.pad(mask, pad_width, 'constant')
        elif fit_to_size:
            raise ValueError(f"Invalid fit_to_size option: {fit_to_size}")
        assert mask.shape[0] >= contour_dims[0] - 1 and mask.shape[1] >= contour_dims[
            1] - 1, "Shifted contour should fit in mask"
        cv2.drawContours(mask, [contour], -1, value, thickness=-1)  # thickness=-1 fills the entire area inside
        # assert np.unique(mask).size > 1, "Cannot return empty (0) mask after contour drawing"
        if np.unique(mask).size <= 1:
            with warnings.catch_warnings():
                warnings.simplefilter("always")  # print this warning each time it occurs
                warnings.warn("Returning empty mask after drawing ...")
        return mask

    def export_tiles(self, layer: Union[str, int], save_dir: Union[str, Path]):
        r"""
        :param layer: layer name or index of contours to extract images for
        :param save_dir: save directory for images
        :return:
        """
        save_dir = Path(save_dir)
        (save_dir/'images').mkdir(exist_ok=True)
        (save_dir/'masks').mkdir(exist_ok=True)
        contours, layer_name = self.annotation.get_layer_points(layer, contour_format=True)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            image = self.get_contour_image(contour, self.reader)
            mask = self.contour_to_mask(contour, shape=(self.tile_size,) * 2 if self.tile_size else ())
            assert image.shape[0:2] == mask.shape[0:2], "Image and mask must be of the same size"
            if self.tile_size is not None:
                images, masks = self.fit_to_size(image, mask)
                for i, (image, mask) in enumerate(zip(images, masks)):
                    name = f'{layer_name}_{x:.d}_{y:.d}_{w:.d}_{h:.d}' + ('' if len(images) == 1 else str(i)) + '.png'
                    imageio.imwrite(save_dir/'images'/name, image)
                    imageio.imwrite(save_dir/'masks'/name, image)

    def fit_to_size(self, image, mask, multitile_threshold=2):
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
                        images.append(image[i:i+self.tile_size, j:j+self.tile_size])
                        masks.append(mask[i:i+self.tile_size, j:j+self.tile_size])
            else:
                images = [self.center_crop(image)]
                masks = [self.center_crop(mask)]
        else:
            images = [image]
            masks = [mask]
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


