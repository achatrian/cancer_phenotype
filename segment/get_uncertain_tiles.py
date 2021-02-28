from pathlib import Path
from argparse import ArgumentParser
import bisect
import random
import numpy as np
from tqdm import tqdm
from openslide.lowlevel import OpenSlideError
from skimage.feature import peak_local_max
from data.images.wsi_reader import WSIReader
from annotation.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--num_tiles', type=int, default=10)
    parser.add_argument('--experiment_name', type=str, default='mpp0.4_active_learn_14f')
    parser.add_argument('--original_annotations_dirname', type=str, default='annotations')
    parser.add_argument('--shuffle_annotations', action='store_true')
    # wsi reader options
    parser.add_argument('--qc_mpp', default=2.0, type=float, help="MPP value to perform quality control on slide")
    parser.add_argument('--mpp', default=0.50, type=float, help="MPP value to read images from slide")
    parser.add_argument('--data_dir', type=str, default='', help="Dir where to save qc result")
    parser.add_argument('--check_tile_blur', action='store_true', help="Check for blur")
    parser.add_argument('--check_tile_fold', action='store_true', help="Check tile fold")
    parser.add_argument('--overwrite_qc', action='store_true', help="Overwrite saved quality control data")
    parser.add_argument('--patch_size', type=int, default=1024, help="Pixel size of patches (at desired resolution)")
    parser.add_argument('--verbose', action='store_true', help="Print more information")
    parser.add_argument('--tissue_threshold', type=float, default=0.4,
                        help="Threshold of tissue filling in tile for it to be considered a tissue tile")
    parser.add_argument('--saturation_threshold', type=int, default=25,
                        help="Saturation difference threshold of tile for it to be considered a tissue tile")
    args = parser.parse_args()

    uncertainty_maps_paths = list((args.data_dir/'data'/'uncertainty_maps').glob('*.tiff'))
    if args.shuffle_annotations:
        random.shuffle(uncertainty_maps_paths)
    uncertainty_annotations_dir = Path(args.data_dir, 'data', 'uncertainty_annotations')
    uncertainty_annotations_dir.mkdir(exist_ok=True, parents=True)
    print(f"Creating uncertainty annotations for {len(uncertainty_maps_paths)} uncertainty maps")
    for uncertainty_map_path in uncertainty_maps_paths:
        uncertainty_map = WSIReader(uncertainty_map_path, args)
        for suffix in ('.ndpi', '.svs', '.tiff'):
            tissue_slide_path = (args.data_dir/uncertainty_map_path.name).with_suffix(suffix)  # TODO does with_suffix cut the slide id?
            if tissue_slide_path.exists():
                break
            else:
                try:
                    tissue_slide_path = next((args.data_dir).glob(f'*/{uncertainty_map_path.with_suffix(suffix).name}'))
                    if tissue_slide_path.exists():
                        break
                except StopIteration:
                    pass
        else:
            raise ValueError(f"No tissue slide for '{uncertainty_map_path.name}'")
        tissue_slide = WSIReader(tissue_slide_path, args)
        try:
            tissue_slide.find_tissue_locations()
        except OpenSlideError:
            print(f"Cannot read from slide '{tissue_slide_path.name}'")
            continue
        peak_uncertainty_locations, peak_uncertainty_values = [], []
        # find locations with highest uncertainty
        for x, y in tqdm(tissue_slide.tissue_locations):
            try:
                tile = np.array(uncertainty_map.read_region((x, y), tissue_slide.read_level, (args.patch_size,) * 2))
            except OpenSlideError:
                break
            tile_peaks = peak_local_max(tile)
            if tile_peaks.size == 0:
                continue
            median_peak = np.median([tile_peaks])
            if len(peak_uncertainty_values) == 0:
                peak_uncertainty_values.append(median_peak)
                peak_uncertainty_locations.append((x, y))
            else:
                position = bisect.bisect_left(peak_uncertainty_values, median_peak)
                peak_uncertainty_values.insert(position, median_peak)
                peak_uncertainty_locations.insert(position, (x, y))
                if len(peak_uncertainty_values) > args.num_tiles:
                    peak_uncertainty_values.pop(0), peak_uncertainty_locations.pop(0)
        else:
            # THIS PART OF CODE ONLY EXECUTES IF THERE IS NO BREAK IN THE CODE ABOVE
            # build annotation with boxes bounding areas of highest uncertainty that need re-annotation
            uncertainty_annotation = AnnotationBuilder(tissue_slide_path.name[:-len(suffix)], 'reannotate',
                                                       ('uncertain',))
            original_annotation_path = Path(args.data_dir, 'data',
                                            args.original_annotations_dirname,
                                            args.experiment_name,
                                            tissue_slide_path.with_suffix('.json').name)
            try:
                original_annotation = AnnotationBuilder.from_annotation_path(original_annotation_path)
                for x, y in peak_uncertainty_locations:
                    uncertainty_annotation.add_item('uncertain', 'rectangle', 'uncertain', filled=True,
                                                    rectangle={
                                                        'x': x,
                                                        'y': y,
                                                        'width': args.patch_size,
                                                        'height': args.patch_size
                                                    })
                uncertainty_annotation = AnnotationBuilder.concatenate(uncertainty_annotation, original_annotation)
            except FileNotFoundError:
                pass  # do not show output of segmentation if it isn't available as a json file
            uncertainty_annotation.dump_to_json(uncertainty_annotations_dir)
    print("Done!")
















