from pathlib import Path
from argparse import ArgumentParser
from copy import deepcopy
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('annotations_dir', type=Path)
    args = parser.parse_args()
    save_dir = args.annotations_dir.parent/('layered_' + args.annotations_dir.name)
    save_dir.mkdir(exist_ok=True, parents=True)
    i = 0
    for i, annotation_path in enumerate(tqdm(list(args.annotations_dir.iterdir()))):
        if not annotation_path.suffix == '.json':
            continue
        annotation = AnnotationBuilder.from_annotation_path(annotation_path)
        layered_annotation = AnnotationBuilder(annotation.slide_id, annotation.project_name)
        for layer_name, layer in annotation.layers.items():
            for item in layer['items']:
                try:
                    decision_label = item['data']['validations'][0]['decision']
                except KeyError as err:
                    continue
                if decision_label not in layered_annotation.layers:
                    layered_annotation.add_layer(decision_label)
                decision_item = deepcopy(item)
                layered_annotation.add_item(decision_label, 'path', item=decision_item)
        layered_annotation.dump_to_json(save_dir)
    print(f"Layered {i} annotations ...")


