from pathlib import Path
import argparse
import json
from math import inf
from random import shuffle
import multiprocessing as mp
from instance_tile_exporter import InstanceTileExporter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--annotations_dirname', type=str, default='annotations')
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.4)
    parser.add_argument('--outer_label', type=str, default='epithelium')
    parser.add_argument('--label_values', type=json.loads, default='[["epithelium", 200], ["lumen", 250]]',
                        help='!!! NB: this would be "[[\"epithelium\", 200], [\"lumen\", 250]]" if passed externally')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max_instances_per_slide', type=int, default=inf)
    parser.add_argument('--tiles_dirname', type=str, default='tiles')
    parser.add_argument('--stop_overwrite', action='store_true')
    parser.add_argument('--shuffle_slides', action='store_true')
    args = parser.parse_args()

    def run_exporter(slide_id):
        try:
            try:
                images_paths = list(Path(args.data_dir, 'data', args.tiles_dirname, args.outer_label, 'slide_id').iterdir())
            except FileNotFoundError:
                images_paths = []
            if args.stop_overwrite and len(images_paths) > 0:
                return
            exporter = InstanceTileExporter(args.data_dir,
                                            slide_id,
                                            experiment_name=args.experiment_name,
                                            annotations_dirname=args.annotations_dirname,
                                            tile_size=args.tile_size,
                                            mpp=args.mpp,
                                            label_values=args.label_values)
            exporter.export_tiles(args.outer_label, args.data_dir / 'data' / args.tiles_dirname,
                                  max_instances=args.max_instances_per_slide)
            print(f"Instance tiles exported from {slide_id}")
        except FileNotFoundError as err:
            print(err)

    slide_ids = [annotation_path.with_suffix('').name
                 for annotation_path in (args.data_dir/'data'/args.annotations_dirname/args.experiment_name).iterdir()
                 if annotation_path.suffix == '.json']

    if args.shuffle_slides:
        shuffle(slide_ids)
    if args.workers:
        with mp.Pool(args.workers) as pool:
            pool.map(run_exporter, slide_ids)
    else:
        for slide_id in slide_ids:
            run_exporter(slide_id)
    print("Done!")
