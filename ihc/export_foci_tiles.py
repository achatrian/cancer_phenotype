from pathlib import Path
import argparse
import json
import multiprocessing as mp
from datetime import datetime
from shutil import rmtree
from data.instance_tile_exporter import InstanceTileExporter
from openslide import OpenSlideError


foci_labels = list(f'Focus{i}' for i in range(1, 9))
label_values = {focus_label: 255 for focus_label in foci_labels}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--save_dirname', type=str, default='tiles')
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--min_read_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.25)
    parser.add_argument('--min_mask_fill', default=0.0, help="minimum focus area contained in tile for tile to be exported")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stop_overwrite', action='store_true')
    parser.add_argument('--annotations_dirname', type=str, default=None)
    parser.add_argument('--set_mpp', default=None, type=float, help="base mpp in images, to be provided in cases the images don't have such metadata")
    parser.add_argument('-ds', '--debug_slide', default=None, action='append')
    parser.add_argument('--start_fresh', action='store_true', help="Delete all existing tiles before writing the new ones")
    args = parser.parse_args()

    def run_exporter(slide_id):
        if args.stop_overwrite and all(Path(args.data_dir, 'data', args.save_dirname, focus_label).is_dir() for focus_label in foci_labels):
            return None
        try:
            exporter = InstanceTileExporter(args.data_dir, slide_id, tile_size=args.tile_size, mpp=args.mpp,
                                            label_values=label_values, annotations_dirname=args.annotations_dirname,
                                            partial_id_match=False, set_mpp=args.set_mpp)
            for focus_label in foci_labels:
                exporter.export_tiles(focus_label, args.data_dir / 'data' / args.save_dirname,
                                      min_read_size=args.min_read_size, min_mask_fill=args.min_mask_fill, smoothing=10)
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
        except FileNotFoundError as err:
            print(f"No image file for id '{slide_id}'")
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

    if args.debug_slide is not None:
        slide_ids = [slide_id for slide_id in slide_ids if slide_id in args.debug_slide]
    if args.start_fresh and not args.debug_slide:
        print("Deleting all pre-existing tiles ...")
        try:
            rmtree(args.data_dir/'data'/args.save_dirname)
        except FileNotFoundError:
            pass
    print("Writing foci tiles ...")
    if args.workers:
        with mp.Pool(args.workers) as pool:
            slide_failures = pool.map(run_exporter, slide_ids)
    else:
        slide_failures = []
        for slide_id in slide_ids:
            slide_failures.append(run_exporter(slide_id))
    with open(args.data_dir/'data'/args.save_dirname/f'failed_image_reads_{str(datetime.now())[:10]}.json', 'w') as failed_image_reads_file:
        json.dump([slide_failure for slide_failure in slide_failures if slide_failure is not None],
                  failed_image_reads_file)
    with open(args.data_dir/'data'/args.save_dirname/f'tiles_info_{str(datetime.now())[:10]}.json', 'w') as tiles_info_reads_file:
        dict_args = {k: v if not isinstance(v, Path) else str(v) for k, v in vars(args).items()}
        json.dump(dict_args, tiles_info_reads_file)
    print("Done!")
    # TODO only applying annotations to CK5 images for a lot of cases -- apply to all!!
