import json
import tqdm
from pathlib import Path

annotation_path = '/mnt/rescomp/projects/TCGA_prostate/TCGA/data/tcga_annnotations.csv'
dest_dir = '/mnt/rescomp/projects/TCGA_prostate/TCGA/data/tumour_area_annotations'
with open(annotation_path, 'r') as annotations_file:
    annotations_file.readline()  # skip first line
    for line in tqdm.tqdm(annotations_file):
        chunks = line.split(',')
        slide_id = chunks[0][1:-1]  # remove extra quotes
        annotator = chunks[-1][1:-2]  # remove extra quotes
        line = ','.join(line.split(',')[1:-1])[1:-1]  # remove extra quotes
        ann_obj = json.loads(line)['annotation']
        ann_obj['slide_id'] = slide_id
        ann_obj['project_name'] = 'TCGA'
        ann_obj['layer_names'] = [layer['name'] for layer in ann_obj['layers']]
        ann_file_path = Path(dest_dir)/(slide_id + '.json')
        with open(ann_file_path, 'w') as ann_file:
            json.dump(ann_obj, ann_file)




