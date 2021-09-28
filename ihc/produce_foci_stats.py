import argparse
from pathlib import Path
import re
import json
from datetime import datetime
from collections import Counter
import imageio
import pandas as pd
import numpy as np
from tqdm import tqdm


r"""Get overview of foci images generated from annotations on IHC Request Cases"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--ihc_data_file', type=Path, default='/well/rittscher/projects/IHC_Request/data/documents/additional_data_2020-02-20.csv')
    args = parser.parse_args()
    tiles_dir = args.data_dir/'data'/'tiles'
    slide_data = pd.read_csv(args.ihc_data_file)
    # build a per slide per focus break down (each row)
    all_data = []
    columns = ['slide_id', 'focus_n', 'staining_code', 'case_type', 'n_foci_in_layer', 'focus_size', 'num_images']
    slides_with_missing_data, focus_extends_across_cuts = set(), {}
    slide_ids, foci_per_slide = set(), {}
    focus_paths = sorted(list(tiles_dir.iterdir()), key=lambda p: p.name)
    foci_stains = {}
    cases_type_count = Counter()
    for focus_path in tqdm(focus_paths, desc="Focus num"):
        if not focus_path.is_dir():
            continue
        i = int(focus_path.name[-1])
        for slide_path in focus_path.iterdir():
            slide_id = slide_path.name
            slide_ids.add(slide_id)
            if slide_id not in foci_per_slide:
                foci_per_slide[slide_id] = []
            try:
                slide_row = next(slide_row for j, slide_row in slide_data.iterrows() if slide_row['Image'] == slide_id)
            except StopIteration:
                slides_with_missing_data.add(slide_id)
                slide_row = None
            image_paths = list(slide_path.glob('*_image.png'))
            if image_paths:
                foci_per_slide[slide_id].append(focus_path.name)
            if slide_row is not None:
                foci_stains[slide_id] = slide_row['Staining code']
            # foci can extend across different adjacent cuts of tissue that are present on the same slide
            # in this case the focus size is computed as an average of sizes over different cuts
            focus_size = {}
            tile_sizes = set()
            for image_path in image_paths:
                mask_path = image_path.parent/image_path.name.replace('image', 'mask')
                mask = imageio.imread(mask_path)
                tile_sizes.add(f'{mask.shape[0]}x{mask.shape[1]}')
                x, y, w, h, component_n = re.match(rf'Focus{i}_(\d+)_(\d+)_(\d+)_(\d+)_(\d*)_image.png', image_path.name).groups()
                if (x, y, w, h) not in focus_size:
                    focus_size[(x, y, w, h)] = 0
                focus_size[(x, y, w, h)] += (mask > 0).sum()  # counting how many pixels in this tile belong to the focus
            if len(focus_size) > 1:
                if slide_id not in focus_extends_across_cuts:
                    focus_extends_across_cuts[slide_id] = []
                focus_extends_across_cuts[slide_id].append(focus_path.name)
            if slide_row is not None:
                if slide_row['Case type'] == 'Real':
                    cases_type_count.update([str(slide_row['Diagnosis'])])
                else:
                    cases_type_count.update(slide_row['Case type'])
            all_data.append(dict(
                slide_id=slide_id, focus_n=i,
                staining_code=slide_row['Staining code'] if slide_row is not None else None,
                case_type=slide_row['Case type'] if slide_row is not None else None,
                n_focus_cuts=len(focus_size),  # how many times a focus is present on one slide
                focus_size=np.mean(tuple(focus_size.values())) if image_paths else 0,
                num_images=len(image_paths),
                image_sizes=';'.join(tile_sizes) if image_paths else ''  # what kind of tile sizes does this focus contain ? (this should always be the same if only one tile_size setting was used to export the images)
            ))
    focus_annotations_stats = pd.DataFrame(all_data, columns=columns)
    focus_annotations_stats = focus_annotations_stats.set_index('slide_id')
    slides_with_no_foci_annotations = [slide_id for slide_id, foci in foci_per_slide.items() if len(foci) == 0]
    timestamp = str(datetime.now())[:10]
    with open(args.data_dir/'data'/f'focus_annotations_stats_{timestamp}.csv', 'w') as focus_stats_file:
        focus_annotations_stats.to_csv(focus_stats_file)
    with open(args.data_dir/'data'/f'focus_annotations_info_{timestamp}.json', 'w') as focus_info_file:
        json.dump({
            'slide_ids': list(slide_ids),
            'num_slides': len(slide_ids),
            'slides_with_missing_data': list(slides_with_missing_data),
            'num_slides_with_missing_data': len(slides_with_missing_data),
            '%slides_with_missing_data': len(slides_with_missing_data)/len(slide_ids),
            'slides_with_no_foci_annotations': list(slides_with_no_foci_annotations),
            'num_slides_with_no_foci_annotations': len(slides_with_no_foci_annotations),
            '%slides_with_no_foci_annotations': len(slides_with_no_foci_annotations)/len(slide_ids),
            'focus_extends_across_cuts': focus_extends_across_cuts,
            'foci_per_slide': foci_per_slide,
            '%stains': dict(Counter(foci_stains.values())),
            '%slides_with_at_least_1_focus': len([slide_id for slide_id, foci in foci_per_slide.items() if len(foci) >= 1])/len(slide_ids),
            '%slides_with_at_least_2_foci': len([slide_id for slide_id, foci in foci_per_slide.items() if len(foci) >= 2])/len(slide_ids),
            '%slides_with_at_least_3_foci': len([slide_id for slide_id, foci in foci_per_slide.items() if len(foci) >= 3])/len(slide_ids),
            '%slides_with_at_least_4_foci': len([slide_id for slide_id, foci in foci_per_slide.items() if len(foci) >= 4])/len(slide_ids),
            'case_type_counts': dict(cases_type_count)
        }, focus_info_file)
    print(f"Done! Saved {args.data_dir/'data'/f'focus_annotations_stats_{timestamp}.csv'} and {args.data_dir/'data'/f'focus_annotations_info_{timestamp}.json'}")


















