from pathlib import Path
from itertools import chain
import json
import csv
import re
from tqdm import tqdm
import numpy as np
import cv2
from base.options.train_options import TrainOptions
from base.utils.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    # FIXME outdated -- copied from tilepheno dataset
    opt = TrainOptions.parse()
    tiles_path = Path(opt.data_dir) / 'data' / 'tiles'
    wsi_paths = [path for path in tiles_path.iterdir() if
                 path.is_dir()]  # one per wsi image the tiles were derived from
    paths = [path for path in chain(*(wsi_path.glob(opt.image_glob_pattern) for wsi_path in wsi_paths))]
    assert paths, "Cannot be empty"
    with open(opt.split_file, 'r') as split_json:
        split = json.load(split_json)
    tqdm.write("Selecting split tiles within annotation area (might take a while) ...")
    opt.phase = opt.phase if opt.phase != 'val' else 'test'  # check on test set during training (TEMP)
    phase_split = set(split[opt.phase])  # ~O(1) __contains__ check through hash table
    id_len = len(phase_split.pop())  # checks length of id
    tqdm.write("Filtering by training split ...")
    paths = sorted(path for path in paths if path.parent.name[:id_len] in phase_split)
    assert paths, "Cannot be empty"
    tumour_annotations_dir = Path(opt.data_dir) / 'data' / 'tumour_area_annotations'
    if tumour_annotations_dir.is_dir():
        # only paths in annotated contours will be used - slides with no annotations are discarded
        paths_in_annotation = []
        annotation_paths = list(tumour_annotations_dir.iterdir())
        tqdm.write("Filtering by tumour area ...")
        for annotation_path in tqdm(annotation_paths):
            with open(annotation_path, 'r') as annotation_file:
                annotation = json.load(annotation_file)
                contours, labels = AnnotationBuilder.from_object(annotation).get_layer_points(0, contour_format=True)
                if len(contours) > 1 and sum(bool(contour.size) for contour in contours) > 1:
                    areas = [cv2.contourArea(contour) for contour in contours if contour.size]
                    tumour_area = contours[np.argmax(areas)]
                else:
                    tumour_area = contours[0]
                if tumour_area.size == 0:  # some contours are empty
                    continue
                for i, path in enumerate(paths):
                    if not annotation['slide_id'] in path.parent.name:
                        continue
                    origin_corner = tuple(int(s.replace('.png', '')) for s in str(path.name).split('_'))
                    opposite_corner = (origin_corner[0] + opt.patch_size, origin_corner[1] + opt.patch_size)
                    if cv2.pointPolygonTest(tumour_area, origin_corner, measureDist=False) \
                            and cv2.pointPolygonTest(tumour_area, opposite_corner, measureDist=False):
                        paths_in_annotation.append(path)
        paths = paths_in_annotation
        split['tiles_file'] = opt.split_file.replace('', '.json') + '_tiles.csv'
        with open(re.sub('.json', '', opt.split_file) + '_tiles.csv', 'w') as split_tiles_file:
            csv.writer(split_tiles_file).writerows(paths)
    else:
        paths = paths
