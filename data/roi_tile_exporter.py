from pathlib import Path
from typing import Union
from numbers import Number
import argparse
import numpy as np
import cv2
import imageio
from tqdm import tqdm
from skimage.filters import gaussian
from image.wsi_reader import WSIReader
from annotation.annotation_builder import AnnotationBuilder
from data import read_annotations, contour_to_mask
from annotation.mask_converter import MaskConverter
from base.utils import debug


class ROITileExporter:
    r"""Extract tiles from an annotation area"""
    def __init__(self, data_dir, slide_id, tile_size=1024, mpp=0.2, max_num_tiles=np.inf,
                 label_values=(('epithelium', 200), ('lumen', 250)), roi_dir_name=None):
        self.data_dir = Path(data_dir)
        self.slide_id = slide_id
        self.tile_size = tile_size
        self.mpp = mpp
        self.max_num_tiles = max_num_tiles
        self.label_values = dict(label_values)
        if roi_dir_name is not None and not (self.data_dir/'data'/roi_dir_name).is_dir():
            ValueError(f"{str(self.data_dir/'data'/roi_dir_name)} is not a directory")
        self.roi_dir = self.data_dir/'data'/roi_dir_name if roi_dir_name is not None else None
        try:
            self.slide_path = next(path for path in self.data_dir.iterdir() if slide_id in path.name
                                   and path.name.endswith(('.svs', 'ndpi', 'tiff')))
        except StopIteration:
            raise ValueError(f"No image file matching slide id: {slide_id}")
        slide_opt = WSIReader.get_reader_options(include_path=False, args={
            'patch_size': tile_size,
            'mpp': mpp,
            'data_dir': str(data_dir)
        })
        self.slide = WSIReader(slide_opt, self.slide_path)
        self.slide.opt.patch_size = self.tile_size
        self.slide.find_tissue_locations()
        self.original_tissue_locations = self.slide.tissue_locations
        assert self.original_tissue_locations, "Cannot have 0 tissue locations"
        self.contour_lib = read_annotations(self.data_dir, slide_ids=(self.slide_id,))[self.slide_id]
        self.tile_size = tile_size
        self.center_crop = CenterCrop(self.tile_size)
        self.bounding_boxes = {
            layer_name: [
                cv2.boundingRect(contour) for contour in contours
            ] for layer_name, contours in self.contour_lib.items()
        }

        if self.roi_dir is not None:
            self.roi_contour_lib = read_annotations(self.roi_dir, (self.slide_id,), True)[self.slide_id]

    @staticmethod
    def get_tile_contours(contours, bounding_rects, labels, tile_rect):
        x, y, w, h = tile_rect
        overlap_labels = {'overlap', 'contained'}
        tile_contours, tile_labels = [], []
        for contour, bounding_rect, label in zip(contours, bounding_rects, labels):
            if contour.size == 0:
                continue
            overlap, origin_rect, areas = AnnotationBuilder.check_relative_rect_positions(tile_rect, bounding_rect)
            if overlap not in overlap_labels:
                continue
            tile_contours.append(contour)
            tile_labels.append(label)
        return tile_contours, tile_labels

    def stitch_segmentation_map(self, tile_contours, labels, mask_origin):
        r"""All contours are assumed to fit into the tile"""
        mask = None
        for contour, label in zip(tile_contours, labels):
            mask = contour_to_mask(
                contour,
                self.label_values[label],
                (self.tile_size, self.tile_size),
                mask,
                mask_origin
            )  # contours that lay outside of mask are cut
        return mask

    def export_tiles(self, area_layer: Union[str, int], save_dir: Union[str, Path], hier_rule=lambda x: x):
        r"""
        :param area_layer: annotation layer marking areas in the slide to extract tiles from
        :param save_dir: where to save the tiles
        :return:
        """
        # TODO externalise parameters
        save_dir = Path(save_dir)/area_layer
        slide_dir = save_dir/self.slide_id
        save_dir.mkdir(exist_ok=True), slide_dir.mkdir(exist_ok=True)
        try:
            areas_to_tile = self.contour_lib[area_layer] if self.roi_dir is None else self.roi_contour_lib[area_layer]
        except KeyError:
            raise KeyError(f"Invalid exporting layer '{area_layer}': no such layer in annotation for {self.slide_id}")
        contours, labels, bounding_rects = [], [], []
        for layer_name, layer_contours in self.contour_lib.items():
            if layer_name == area_layer:
                continue  # get all contours except those delimiting the export area
            contours.extend(layer_contours)
            bounding_rects.extend(self.bounding_boxes[layer_name])
            labels.extend([layer_name] * len(layer_contours))
        self.slide.filter_locations(areas_to_tile)
        print("Extracting tiles and masks ...")
        num_saved_images = 0
        value_hier = sorted(self.label_values.values(), key=hier_rule)
        converter = MaskConverter(value_hier=value_hier)
        for x, y in tqdm(self.slide.tissue_locations):
            tile = self.slide.read_region((x, y))
            tile_contours, tile_labels = self.get_tile_contours(contours, bounding_rects, labels, (x, y, self.tile_size, self.tile_size))
            if not tile_contours:
                continue
            mask = self.stitch_segmentation_map(tile_contours, tile_labels, (x, y))
            tile = np.array(tile, dtype=np.uint8)
            mask = cv2.dilate(mask, np.ones((3, 3)))  # pre-dilate to remove jagged boundary from low-res contour extraction
            value_binary_masks = []
            for value in value_hier:
                value_binary_mask = converter.threshold_by_value(value, mask)
                value_binary_mask = (gaussian(value_binary_mask, sigma=15) > 0.5).astype(np.uint8) # smoothen jagged edges TODO these parameters are very empirical?
                value_binary_mask = converter.remove_ambiguity(value_binary_mask)
                value_binary_masks.append(value_binary_mask)
            mask = np.zeros_like(mask)
            for value_binary_mask, value in zip(value_binary_masks, value_hier):
                mask[value_binary_mask > 0] = value
            mask = np.array(mask, dtype=np.uint8)
            assert tile.shape[:2] == mask.shape, f"Tile and mask shapes don't match: {tile.shape[:2]} != {mask.shape}"
            imageio.imwrite(save_dir/f'{x}_{y}_image.png', tile)
            imageio.imwrite(save_dir/f'{x}_{y}_mask.png', mask)
            num_saved_images += 1
        self.slide.tissue_locations = self.original_tissue_locations  # restore full list for slide re-use
        print(f"Saved {num_saved_images}x2 images. Done!")


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
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('slide_id', type=str)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.2)
    parser.add_argument('--max_num_tiles', default=np.int)
    parser.add_argument('--area_label', type=str, default='Tumour area')
    parser.add_argument('--roi_dir_name', default=None)
    args = parser.parse_args()
    exporter = ROITileExporter(args.data_dir,
                               args.slide_id,
                               args.tile_size,
                               args.mpp,
                               args.max_num_tiles,
                               roi_dir_name=args.roi_dir_name)
    exporter.export_tiles(args.area_label, args.data_dir/'data'/'tiles')


