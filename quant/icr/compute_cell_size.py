import argparse
from pathlib import Path
from datetime import datetime
import multiprocessing as mp
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from data.contours import read_annotations, check_point_overlap


def get_nuclear_area(nucleus_contour, tumour_area_contour):
    if nucleus_contour.size == 0 or nucleus_contour.shape[0] < 3:
        return np.nan
    return cv2.contourArea(nucleus_contour) if check_point_overlap(tumour_area_contour, nucleus_contour) else np.nan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    # tumour area contours are in annotations
    tumour_area_struct = read_annotations(args.data_dir, annotation_dirname='tumour_area_annotations')
    columns = ['image', 'tumour_roi', 'tumour_area', 'nuclear_size:mean', 'nuclear_size:var', 'nuclear_size:median',
               'num_nuclei', 'cell_size:mean', 'cell_size:median']
    cell_size_data = []
    for slide_id in tqdm(tumour_area_struct, desc='slide'):
        tumour_area_contours = tumour_area_struct[slide_id]['Tumour area']
        nucleus_contours = read_annotations(args.data_dir, slide_ids=(slide_id,),
                                            annotation_dirname='nuclei_annotations')[slide_id]['nuclei']
        for tumour_area_contour in tqdm(tumour_area_contours, desc='roi'):
            x, y, w, h = cv2.boundingRect(tumour_area_contour)  # FIXME
            roi_id = f'{x}_{y}_{w}_{h}'
            nuclear_areas = []
            if args.workers:
                with mp.Pool(args.workers) as pool:
                    nuclear_areas = np.array(pool.starmap(get_nuclear_area, [
                        nucleus_contours,
                        [(x, y, w, h)]*len(nucleus_contours)
                    ]))
                    nuclear_areas = nuclear_areas[~np.isnan(nuclear_areas)]
            else:
                for nucleus_contour in nucleus_contours:
                    if nucleus_contour.size == 0 or nucleus_contour.shape[0] < 3:
                        continue
                    if not check_point_overlap(tumour_area_contour, nucleus_contour, 3):
                        continue
                    nuclear_areas.append(cv2.contourArea(nucleus_contour))
            nuclear_areas = np.array(nuclear_areas)
            tumour_area = cv2.contourArea(tumour_area_contour)
            nuclear_mean, nuclear_median, nuclear_var = np.mean(nuclear_areas), np.median(nuclear_areas), np.var(nuclear_areas)
            tumour_data = {
                'image': slide_id,
                'tumour_roi': roi_id,
                'tumour_area': tumour_area,
                'nuclear_size:mean': nuclear_mean,
                'nuclear_size:median': nuclear_median,
                'nuclear_size:var': nuclear_var,
                'num_nuclei': nuclear_areas.size,
                'cell_size:mean': tumour_area/nuclear_mean,
                'cell_size:median': tumour_area/nuclear_median,
            }
            cell_size_data.append(tumour_data)
    cell_size_frame = pd.DataFrame(cell_size_data, columns=columns)
    cell_size_frame.to_csv(args.data_dir/'data'/f'cell_size_{str(datetime.now())[:10]}.csv')
    # nuclei in glands
    # glands_struct = read_annotations(args.data_dir, annotation_dirname='glands_annotations')










