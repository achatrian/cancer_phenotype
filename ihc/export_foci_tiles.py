from pathlib import Path
import argparse
import json
import multiprocessing as mp
from datetime import datetime

from data.instance_tile_exporter import InstanceTileExporter
from openslide import OpenSlideError


foci_labels = list(f'Focus{i}' for i in range(1, 9))
label_values = {focus_label: 255 for focus_label in foci_labels}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.25)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stop_overwrite', action='store_true')
    parser.add_argument('--annotations_dirname', type=str, default=None)
    args = parser.parse_args()

    def run_exporter(slide_id):
        if args.stop_overwrite and all(Path(args.data_dir, 'data', 'tiles', focus_label).is_dir() for focus_label in foci_labels):
            return None
        try:
            exporter = InstanceTileExporter(args.data_dir,
                                            slide_id,
                                            args.tile_size,
                                            args.mpp,
                                            label_values,
                                            args.annotations_dirname,
                                            partial_id_match=False)
            for focus_label in foci_labels:
                exporter.export_tiles(focus_label, args.data_dir/'data'/'tiles', min_mask_fill=0.0)
        except OpenSlideError as err:
            print(f"Image {slide_id} cannot be read with openslide, it may be corrupted - err: {err}")
            return {
                'slide_id': slide_id,
                'error_message': str(err)
            }
        except ValueError as err:
            print(err)
            return {
                'slide_id': slide_id,
                'error_message': str(err)
            }
        print(f"Focus tiles exported from {slide_id}")
        return None

    slide_ids = [annotation_path.with_suffix('').name
                 for annotation_path in
                 (args.data_dir/'data'/(args.annotations_dirname if args.annotations_dirname is not None else 'annotations')).iterdir()
                 if annotation_path.suffix == '.json']

    if args.workers:
        with mp.Pool(args.workers) as pool:
            slide_failures = pool.map(run_exporter, slide_ids)
    else:
        slide_failures = []
        for slide_id in slide_ids:
            slide_failures.append(run_exporter(slide_id))
    with open(args.data_dir/f'failed_image_reads_{str(datetime.now())[:10]}.json', 'w') as failed_image_reads_file:
        json.dump([slide_failure for slide_failure in slide_failures if slide_failure is not None],
                  failed_image_reads_file)
    print("Done!")
    # TODO only applying annotations to CK5 images for a lot of cases -- apply to all!!
