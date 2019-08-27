from pathlib import Path
import argparse
import json
import multiprocessing as mp
from instance_tile_exporter import InstanceTileExporter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.4)
    parser.add_argument('--outer_label', type=str, default='epithelium')
    parser.add_argument('--label_values', type=json.loads, default='[["epithelium", 200], ["lumen", 250]]',
                        help='!!! NB: this would be "[[\"epithelium\", 200], [\"lumen\", 250]]" if passed externally')
    parser.add_argument('--roi_dir_name', default='tumour_area_annotations')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--stop_overwrite', action='store_true')
    args = parser.parse_args()
    dir_names = set(path.name for path in (args.data_dir/'data'/'tiles').iterdir())

    def run_exporter(slide_id):
        if args.stop_overwrite and slide_id not in dir_names:
            return
        exporter = InstanceTileExporter(args.data_dir,
                                        slide_id,
                                        args.tile_size,
                                        args.mpp,
                                        args.label_values)
        exporter.export_tiles(args.area_label, args.data_dir / 'data' / 'tiles')
        print(f"ROI tiles exported from {slide_id}")

    slide_ids = [annotation_path.with_suffix('').name
                 for annotation_path in (args.data_dir/'data'/'annotations').iterdir()
                 if annotation_path.suffix == '.json']

    if args.workers:
        with mp.Pool(args.workers) as pool:
            pool.map(run_exporter, slide_ids)
    else:
        for slide_id in slide_ids:
            run_exporter(slide_id)
    print("Done!")



