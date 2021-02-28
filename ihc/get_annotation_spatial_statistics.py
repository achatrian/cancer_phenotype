from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import cv2
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder

#/Users/andreachatrian/Mounts/rescomp/gpfs2


def get_annotation_statistics(latest_annotation_path, latest_data_path, name=''):
    annotations = pd.read_csv(latest_annotation_path)
    annotation_data = pd.read_csv(latest_data_path)
    slides_annotators = {}
    slides_annotated_areas = {}
    control_slides = set(row['Image'] for index, row in annotation_data.iterrows() if row['Case type'] == 'Control')
    for index, row in tqdm(annotations.iterrows(), total=len(annotations)):
        slide_id = row['Image']
        if slide_id in control_slides:
            continue
        if slide_id.endswith(('CK5', 'CK5_1', 'CK5_2', 'CK5_3', 'CK4', 'CK3', 'CKAE13', 'panCK',
                              'Other', 'Staining code', '34BE12', '34B12')):
            continue
        annotation = AnnotationBuilder.from_object(json.loads(row['Annotation']))
        total_annotated_area = 0.0
        for layer_name in annotation.layers:
            contours, _ = annotation.get_layer_points(layer_name, contour_format=True)
            total_annotated_area += sum(cv2.contourArea(contour) for contour in contours if contour.size > 0)
        slides_annotated_areas[slide_id] = total_annotated_area
        slides_annotators[slide_id] = row['Annotator']
    total_area_per_annotator = {f'{annotator}_area': sum(area for slide_id, area in slides_annotated_areas.items()
                                                         if slides_annotators[slide_id] == annotator)
                                for annotator in set(slides_annotators.values())}
    annotators = {
        annotator: [(slide_id, slides_annotated_areas[slide_id]) for slide_id, annotator_ in slides_annotators.items()
                    if annotator_ == annotator]
        for annotator in set(slides_annotators.values())}
    spatial_stats = {
        'annotated_areas': slides_annotated_areas,
        **total_area_per_annotator,
        **annotators
    }
    with open(latest_annotation_path.parent / f'annotations_spatial_statistics_{str(datetime.now())[:10]}{"_" + name if name else ""}.json', 'w') \
            as spatial_statistics_file:
        json.dump(spatial_stats, spatial_statistics_file)


if __name__ == '__main__':
    latest_annotation_path = Path("/well/rittscher/projects/IHC_Request/data/documents/annotations_2020-04-21.csv")
    latest_data_path = Path("/well/rittscher/projects/IHC_Request/data/documents/additional_data_2020-04-21.csv")
    richard_val_annotation_path = Path("/well/rittscher/projects/IHC_Request/data/documents/final_validation_annotations/richards_annotations.csv")
    richard_val_data_path = Path("/well/rittscher/projects/IHC_Request/data/documents/validation_data_richard.csv")
    clare_val_annotation_path = Path("/well/rittscher/projects/IHC_Request/data/documents/final_validation_annotations/clares_annotations.csv")
    clare_val_data_path = Path("/well/rittscher/projects/IHC_Request/data/documents/validation_data_clare.csv")
    lisa_val_annotation_path = Path("/well/rittscher/projects/IHC_Request/data/documents/final_validation_annotations/lisas_annotations.csv")
    lisa_val_data_path = Path("/well/rittscher/projects/IHC_Request/data/documents/validation_data_lisa.csv")
    get_annotation_statistics(latest_annotation_path, latest_data_path)
    get_annotation_statistics(clare_val_annotation_path, clare_val_data_path, 'CLARE')
    get_annotation_statistics(richard_val_annotation_path, richard_val_data_path, 'RICHARD')
    get_annotation_statistics(lisa_val_annotation_path, lisa_val_data_path, 'LISA')

