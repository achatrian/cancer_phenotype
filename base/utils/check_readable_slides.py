from pathlib import Path
import re
import json
from datetime import datetime
from tqdm import tqdm
from openslide import OpenSlideError
from argparse import ArgumentParser
from data.images.wsi_reader import WSIReader


r"""Utility that attempts to read all slides of openslide-compatible formats in dir and reports on reading failures"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--set_mpp', type=float, default=None)
    set_mpp = parser.parse_known_args()[0].set_mpp
    opt = WSIReader.get_reader_options(False, False)
    opt.overwrite_qc = True
    image_paths = []
    image_paths += list(path for path in Path(opt.data_dir).glob('*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*.tiff'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.ndpi'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.svs'))
    image_paths += list(path for path in Path(opt.data_dir).glob('*/*.tiff'))
    failure_log = []
    for image_path in tqdm(image_paths, desc='slide'):
        slide_id = re.sub(r'\.(ndpi|svs|tiff)', '', image_path.name)
        print(f"Processing slide: {slide_id} (extension: {image_path.suffix})")
        try:
            slide = WSIReader(image_path, opt, set_mpp)
            slide.find_tissue_locations()
        except OpenSlideError as err:
            print(err)
            failure_log.append({
                'slide_id': slide_id,
                'file': str(image_path),
                'error': str(err),
                'message': f"Error occurred when applying network to slide {slide_id}"
            })
    logs_dir = Path(opt.data_dir, 'data', 'logs')
    logs_dir.mkdir(exist_ok=True, parents=True)
    with open(logs_dir/f'openslide_read_failures_{str(datetime.now())[:10]}', 'w') as failures_log_file:
        json.dump(failure_log, failures_log_file)





