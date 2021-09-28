from pathlib import Path
import argparse
import json
import multiprocessing as mp
import numpy as np
from roi_tile_exporter import ROITileExporter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--annotations_dirname', type=str, default='annotations')
    parser.add_argument('--tile_size', type=int, default=1024)
    parser.add_argument('--mpp', type=float, default=0.4)
    parser.add_argument('--set_mpp', type=float, default=0.25)
    parser.add_argument('--max_num_tiles', default=np.inf)
    parser.add_argument('--area_label', type=str, default='Tumour area')
    # parser.add_argument('--label_values', type=json.loads, default='[["epithelium", 200], ["lumen", 250]]',
    #                     help='!!! NB: this would be "[[\"epithelium\", 200], [\"lumen\", 250]]" if passed externally')
    parser.add_argument('--roi_dir_name', default='tumour_area_annotations')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--tiles_dirname', type=str, default='tiles')
    parser.add_argument('--stop_overwrite', action='store_true')
    parser.add_argument('-ds', '--debug_slide', default=None, action='append')
    args = parser.parse_args()
    try:
        # get label directory
        dir_path = next(path for path in (args.data_dir/'data'/args.tiles_dirname).iterdir()
                        if path.is_dir() and args.area_label in path.name)
    except (StopIteration, FileNotFoundError):
        dir_path = None

    def run_exporter(slide_id):
        if args.stop_overwrite and dir_path is not None and not (dir_path/slide_id).is_dir():
            return
        try:
            exporter = ROITileExporter(args.data_dir, slide_id,
                                       args.experiment_name,
                                       tile_size=args.tile_size,
                                       mpp=args.mpp,
                                       annotations_dirname=args.annotations_dirname,
                                       roi_dir_name=args.roi_dir_name,
                                       set_mpp=args.set_mpp)
            print(f"Exporting tiles from {slide_id} ...")
            exporter.export_tiles(args.area_label, args.data_dir / 'data' / args.tiles_dirname)
            print(f"ROI tiles exported from {slide_id}")
        except KeyError as err:
            print(err)
            print(f"Could not export ROI tiles from {slide_id}")
        except ValueError as err:
            print(slide_id)
            print(err)

    annotation_dir = args.data_dir / 'data' / args.annotations_dirname
    if args.experiment_name is not None:
        annotation_dir /= args.experiment_name
    slide_ids = [annotation_path.with_suffix('').name
                 for annotation_path in annotation_dir.iterdir()
                 if annotation_path.suffix == '.json']

    if args.debug_slide is not None:
        slide_ids = [slide_id for slide_id in slide_ids if slide_id in args.debug_slide]
    if args.workers:
        with mp.Pool(args.workers) as pool:
            pool.map(run_exporter, slide_ids)
    else:
        for slide_id in slide_ids:
            run_exporter(slide_id)
    print("Done!")

