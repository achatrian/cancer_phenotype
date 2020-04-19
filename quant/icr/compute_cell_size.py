import argparse
from pathlib import Path
from datetime import datetime
import csv
import multiprocessing as mp
import numpy as np
import h5py
import cv2
import pandas as pd
from tqdm import tqdm
from data.contours import read_annotations, check_point_overlap


def get_nuclear_area(nucleus_contour, tumour_area_contour):
    if nucleus_contour.size == 0 or nucleus_contour.shape[0] < 3:
        return np.nan
    return cv2.contourArea(nucleus_contour) if check_point_overlap(tumour_area_contour, nucleus_contour, 3) else np.nan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--min_area_size', type=float, default=10000.0)
    args = parser.parse_args()
    # tumour area contours annotated by physician
    tumour_annotations_struct = read_annotations(args.data_dir, annotation_dirname='tumour_area_annotations')
    # tumour areas detected by deep learning
    tumour_areas_struct = read_annotations(args.data_dir, annotation_dirname='tumour_areas')
    columns = ['image', 'tumour_roi', 'tumour_area', 'nuclear_size:mean', 'nuclear_size:var', 'nuclear_size:median',
               'num_nuclei', 'cell_size:mean', 'cell_size:median']
    missing_slides = []
    missing_areas = []
    for slide_id in tumour_annotations_struct:
        if slide_id not in tumour_areas_struct:
            continue
        if 'Tumour area' not in tumour_areas_struct[slide_id]:
            missing_areas.append(slide_id)
            tumour_areas_struct[slide_id]['Tumour area'] = next(iter(tumour_areas_struct[slide_id].values()))
    print(f"Slides with missing areas: {missing_areas}")
    with open(args.data_dir / 'data' / f'cell_size_{str(datetime.now())[:10]}.csv', mode='w+') as temp_data_file:
        with h5py.File(args.data_dir / 'data' / 'cell_sizes.h5', 'w') as areas_file:
            # initialising the pool once only speeds up the feature computation considerably
            with mp.Pool(args.workers) as pool:
                writer = csv.DictWriter(temp_data_file, fieldnames=columns)
                for slide_id in tqdm(tumour_annotations_struct,
                                     desc='slide'):  # NB only processing manually annotated slides
                    # read in manual, rough annotations for tumour areas
                    tumour_annotations_contours = tumour_annotations_struct[slide_id]['Tumour area']
                    try:
                        # read in more accurate, automatic tumour segmentation areas
                        # if slide_id not in tumour_areas_struct:
                        #     raise ValueError(f"Missing areas for {slide_id}")
                        tumour_areas_contours = tumour_areas_struct[slide_id]['tumour regions']
                    except KeyError as err:
                        try:
                            tumour_areas_contours = tumour_areas_struct[slide_id]['Tumour area']
                        except KeyError as err:
                            print(f"No segmentation for annotation - {err} - slide id: {slide_id}")
                            missing_slides.append(slide_id)
                            continue
                    nucleus_contours = read_annotations(args.data_dir, slide_ids=(slide_id,),
                                                        annotation_dirname='nuclei_annotations')[slide_id]['nuclei']
                    # filter tumour area contours by the number of samples
                    tumour_areas_contours = tuple(tumour_area_contour for tumour_area_contour in tumour_areas_contours
                                                  if tumour_area_contour.size > 0 and tumour_area_contour.shape[0] >= 3 and
                                                  cv2.contourArea(tumour_area_contour) > args.min_area_size)
                    for tumour_area_contour in tqdm(tumour_areas_contours, desc='area'):
                        if not any(check_point_overlap(tumour_annotation_contour, tumour_area_contour)
                                   for tumour_annotation_contour in tumour_annotations_contours
                                   if tumour_annotation_contour.size > 0):
                            # only look at areas inside annotations
                            continue
                        x, y, w, h = cv2.boundingRect(tumour_area_contour)
                        roi_id = f'{x}_{y}_{w}_{h}'
                        nuclear_areas = []
                        if args.workers:
                            nuclear_areas = np.array(pool.starmap(get_nuclear_area,
                                                                  [(nucleus, tumour) for nucleus, tumour in
                                                                   zip(nucleus_contours,
                                                                       [tumour_area_contour] * len(nucleus_contours))]
                                                                  ))
                            nuclear_areas = nuclear_areas[~np.isnan(nuclear_areas)]
                        else:
                            for nucleus_contour in nucleus_contours:
                                if nucleus_contour.size == 0 or nucleus_contour.shape[0] < 3:
                                    continue
                                if not check_point_overlap(tumour_area_contour, nucleus_contour, 3):
                                    continue
                                nuclear_areas.append(cv2.contourArea(nucleus_contour))
                        nuclear_areas = np.array(nuclear_areas)
                        print(nuclear_areas.shape)
                        tumour_area = cv2.contourArea(tumour_area_contour)
                        nuclear_mean, nuclear_median, nuclear_var = np.mean(nuclear_areas), np.median(
                            nuclear_areas), np.var(nuclear_areas)
                        tumour_data = {
                            'image': slide_id,
                            'tumour_roi': roi_id,
                            'tumour_area': tumour_area,
                            'nuclear_size:mean': nuclear_mean,
                            'nuclear_size:median': nuclear_median,
                            'nuclear_size:var': nuclear_var,
                            'num_nuclei': nuclear_areas.size,
                            'cell_size': tumour_area / nuclear_areas.size
                        }
                        writer.writerow(tumour_data)
                        dataset = areas_file.create_dataset(f'{slide_id}:{roi_id}', data=nuclear_areas)
                        for k, v in tumour_data.items():
                            dataset.attrs[k] = v
                    print(f"Slide id processed: {slide_id}")
        temp_data_file.seek(0)  # rewind file pointer to beginning
        cell_size_data = list(csv.DictReader(temp_data_file, fieldnames=columns))
    cell_size_frame = pd.DataFrame(cell_size_data, columns=columns)
    processed_slide_ids = set(cell_size_frame['image'])
    print(f"Saving frame with regions from {len(processed_slide_ids)} slides (out of {len(tumour_annotations_struct)}")
    if missing_slides:
        print(f"Missing slides: {missing_slides}")
    cell_size_frame.to_csv(args.data_dir / 'data' / f'cell_size_{str(datetime.now())[:10]}.csv')
    print("Done!")
    # nuclei in glands
    # glands_struct = read_annotations(args.data_dir, annotation_dirname='glands_annotations')
