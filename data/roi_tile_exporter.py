from pathlib import Path
from typing import Union
from numbers import Number
import argparse
import warnings
import json
from datetime import datetime
from collections import OrderedDict
from itertools import chain
import numpy as np
import cv2
import imageio
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.color import rgba2rgb
from data.images.wsi_reader import make_wsi_reader
from data.contours import read_annotations, check_relative_rect_positions, contour_to_mask
from data.contours.instance_masker import InstanceMasker
from annotation.mask_converter import MaskConverter


# TODO test


class ROITileExporter:
    r"""Extract tiles from an annotation area"""
    def __init__(self, data_dir, slide_id, experiment_name, annotations_dirname,
                 tile_size=1024, mpp=0.2, roi_dir_name=None, sigma_smooth=10, set_mpp=None):
        self.data_dir = Path(data_dir)
        self.slide_id = slide_id
        self.experiment_name = experiment_name
        self.converter = MaskConverter()
        self.tile_size = tile_size
        self.mpp = mpp
        self.sigma_smooth = sigma_smooth
        if roi_dir_name is not None and not (self.data_dir/'data'/roi_dir_name).is_dir():
            ValueError(f"{str(self.data_dir/'data'/roi_dir_name)} is not a directory")
        self.roi_dir = self.data_dir/'data'/roi_dir_name if roi_dir_name is not None else None
        try:  # compatible openslide formats: {.tiff|.svs|.ndpi}
            self.slide_path = next(
                chain(self.data_dir.glob(f'{self.slide_id}.tiff'), self.data_dir.glob(f'*/{self.slide_id}.tiff'),
                      self.data_dir.glob(f'{self.slide_id}.svs'), self.data_dir.glob(f'*/{self.slide_id}.svs'),
                      self.data_dir.glob(f'{self.slide_id}.ndpi'), self.data_dir.glob(f'*/{self.slide_id}.ndpi'),
                      self.data_dir.glob(f'{self.slide_id}.isyntax'), self.data_dir.glob(f'*/{self.slide_id}.isyntax'))
            )
        except StopIteration:
            cases_dir = self.data_dir/'cases'  # TODO finish
            try:
                self.slide_path = next(
                    chain(cases_dir.glob(f'{self.slide_id}.tiff'), cases_dir.glob(f'*/{self.slide_id}.tiff'),
                          cases_dir.glob(f'{self.slide_id}.svs'), cases_dir.glob(f'*/{self.slide_id}.svs'),
                          cases_dir.glob(f'{self.slide_id}.ndpi'), cases_dir.glob(f'*/{self.slide_id}.ndpi'),
                          cases_dir.glob(f'{self.slide_id}.isyntax'), cases_dir.glob(f'*/{self.slide_id}.isyntax'))
                )
            except StopIteration:
                raise FileNotFoundError(f"No image file matching id: {slide_id} ")
            # else:
            #     raise FileNotFoundError(f"No image file matching id: {slide_id} (partial id matching: OFF)")
        self.slide = make_wsi_reader(self.slide_path, {
            'patch_size': tile_size,
            'mpp': mpp,
            'data_dir': str(data_dir)
        }, set_mpp=set_mpp)
        self.slide.opt.patch_size = self.tile_size
        self.slide.find_tissue_locations(0.3, 20)
        self.original_tissue_locations = self.slide.tissue_locations
        assert self.original_tissue_locations, "Cannot have 0 tissue locations"
        self.contour_lib = read_annotations(self.data_dir, slide_ids=(self.slide_id,),
                                            annotation_dirname=annotations_dirname,
                                            experiment_name=experiment_name)[self.slide_id]
        # clean contours
        for layer_name in self.contour_lib:
            self.contour_lib[layer_name] = tuple(contour for contour in self.contour_lib[layer_name]
                                                 if contour.size > 0 and contour.shape[0] > 2 and contour.ndim == 3)
        self.tile_size = tile_size
        self.center_crop = CenterCrop(self.tile_size)
        if self.roi_dir is not None:
            self.roi_contour_lib = read_annotations(self.roi_dir, (self.slide_id,), full_path=True)[self.slide_id]
        self.masker = None

    @staticmethod
    def get_tile_contours(contours, bounding_rects, labels, tile_rect):
        overlap_labels = {'overlap', 'contained'}
        tile_contours, tile_labels = [], []
        for contour, bounding_rect, label in zip(contours, bounding_rects, labels):
            if contour.size == 0:
                continue
            overlap, origin_rect, areas = check_relative_rect_positions(tile_rect, bounding_rect)
            if overlap not in overlap_labels:
                continue
            tile_contours.append(contour)
            tile_labels.append(label)
        return tile_contours, tile_labels

    def make_segmentation_map(self, tile_contours, labels, mask_origin):
        r"""All contours are assumed to fit into the tile"""
        mask = None
        shape = (self.tile_size*round(self.mpp/self.slide.mpp_x),
                 self.tile_size*round(self.mpp/self.slide.mpp_y))
        # reorder contours and labels so that innermost is applied last
        order = tuple(self.converter.label_value_map)  # get the label names
        ordering = tuple(order.index(label) for label in labels)
        tile_contours = tuple(contour for _, contour in sorted(zip(ordering, tile_contours), key=lambda oc: oc[0]))
        labels = tuple(label for _, label in sorted(zip(ordering, labels)))
        for contour, label in zip(tile_contours, labels):
            # FIXME must paint in order of labelling epithelium and then lumen!!
            try:
                mask = contour_to_mask(
                    contour, value=self.converter.label_value_map[label], shape=shape, mask=mask, mask_origin=mask_origin
                )  # contours that lay outside of mask are cut
            except ValueError as err:
                if not err.args[0].startswith('Contour'):
                    raise
                else:
                    print("Error while stitching mask:\n###")
                    print(err)
                    print("###")
        return mask

    def export_tiles(self, area_layer: Union[str, int], save_dir: Union[str, Path]):
        r"""
        :param area_layer: annotation layer marking areas in the slide to extract tiles from
        :param save_dir: where to save the tiles
        :return:
        """
        # TODO externalise parameters
        self.masker = InstanceMasker(self.contour_lib, area_layer, self.converter.label_value_map)
        save_dir = Path(save_dir)/area_layer
        slide_dir = save_dir/self.slide_id
        save_dir.mkdir(exist_ok=True, parents=True), slide_dir.mkdir(exist_ok=True, parents=True)
        try:
            areas_to_tile = self.contour_lib[area_layer] if self.roi_dir is None else self.roi_contour_lib[area_layer]
        except KeyError:
            raise KeyError(f"Invalid exporting layer '{area_layer}': no such layer in annotation for {self.slide_id}")
        contours, labels, bounding_rects = [], [], []
        for layer_name, layer_contours in self.contour_lib.items():
            if layer_name == area_layer:
                continue  # get all contours except those delimiting the export area
            contours.extend(layer_contours)
            bounding_rects.extend([cv2.boundingRect(contour) for contour in layer_contours])
            labels.extend([layer_name] * len(layer_contours))
        # remove all tissue locations outside of ROI
        initial_length = len(self.slide.tissue_locations)
        areas_to_tile = [area_contour for area_contour in areas_to_tile if area_contour.size > 0]
        if len(areas_to_tile) == 0:
            warnings.warn("No ROIs to tile ... returning")
            return
        self.slide.filter_locations(areas_to_tile)
        if len(self.slide.tissue_locations) == initial_length:
            warnings.warn("ROI is whole slide image")
        print("Extracting tiles and masks ...")
        num_saved_images = 0
        x_tile_size = self.tile_size*round(self.mpp/self.slide.mpp_x)
        y_tile_size = self.tile_size*round(self.mpp/self.slide.mpp_y)
        for x, y in tqdm(self.slide.tissue_locations):
            tile_contours, tile_labels = self.get_tile_contours(contours, bounding_rects, labels,
                                                                (x, y, x_tile_size, y_tile_size))
            if not tile_contours:
                continue
            mask = self.make_segmentation_map(tile_contours, tile_labels, (x, y))
            mask = cv2.dilate(mask, np.ones((3, 3)))  # pre-dilate to remove jagged boundary from low-res contour extraction
            value_binary_masks = {}
            for label, interval in self.converter.label_interval_map.items():
                if label == 'background':
                    continue  # don't extract contours for background
                value_binary_mask = self.converter.threshold_by_interval(mask, interval)
                if self.sigma_smooth > 0:
                    value_binary_mask = (gaussian(value_binary_mask, sigma=self.sigma_smooth) > 0.5).astype(np.uint8)  # smoothen jagged edges
                # value_binary_mask = converter.remove_ambiguity(
                #     value_binary_mask,
                #     small_object_size=0,  # no need to remove small objects from annotation
                #     final_closing_size=0,  # no need for large closing of annotation images
                #     final_dilation_size=3
                # )
                value_binary_masks[label] = value_binary_mask
            mask = np.zeros_like(mask)
            for label, value in self.converter.label_value_map.items():
                if label == 'background':
                    continue  # don't extract contours for background
                mask[value_binary_masks[label] > 0] = value
            mask = np.array(mask, dtype=np.uint8)
            tile = np.array(self.slide.read_region((x, y), level=self.slide.read_level, size=self.tile_size),
                            dtype=np.uint8)
            if tile.shape[-1] == 4:  # assume tile is in RGBA format
                tile = (rgba2rgb(tile)*255).astype(np.uint8)
            # resize mask according to mpp difference
            mask = cv2.resize(mask, tile.shape[:2], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            assert tile.shape[:2] == mask.shape, f"Tile and mask shapes don't match: {tile.shape[:2]} != {mask.shape}"
            imageio.imwrite(slide_dir/f'{x}_{y}_image.png', tile)
            imageio.imwrite(slide_dir/f'{x}_{y}_mask.png', mask)
            num_saved_images += 1
        self.slide.tissue_locations = self.original_tissue_locations  # restore full list for slide re-use
        with open(slide_dir/'tile_export_info.json', 'w') as info_file:
            json.dump({
                'date': str(datetime.now()),
                'mpp': self.mpp,
                'tile_size': self.tile_size,
                'sigma_smooth': self.sigma_smooth,
                'label_values': str(self.converter.label_value_map)
            }, info_file)
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
    parser.add_argument('--mpp', type=float, default=0.4)
    parser.add_argument('--label_values', type=json.loads, default='[["epithelium", 200], ["lumen", 250]]',
                        help='!!! NB: this would be "[[\"epithelium\", 200], [\"lumen\", 250]]" if passed externally')
    parser.add_argument('--area_label', type=str, default='Tumour area')
    parser.add_argument('--roi_dir_name', default='tumour_area_annotations')
    args = parser.parse_args()
    exporter = ROITileExporter(args.data_dir, args.slide_id, tile_size=args.tile_size, mpp=args.mpp,
                               label_values=args.label_values, roi_dir_name=args.roi_dir_name)
    exporter.export_tiles(args.area_label, args.data_dir/'data'/'tiles')
