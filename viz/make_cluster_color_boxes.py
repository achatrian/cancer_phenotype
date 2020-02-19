from pathlib import Path
import argparse
import colorsys  # output colors have values from 0 to 1 (but 0-255 space is accepted for RGB)
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from annotation.annotation_builder import AnnotationBuilder

r"""Given a certain clustering of objects belonging to a slide set, 
create an AIDA annotation with coloured bounding boxes over those slides"""

# COLORBREW qualitative colormap with 12 classes http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
rgb_colormap = ['rgb(166,206,227)', 'rgb(31,120,180)', 'rgb(178,223,138)', 'rgb(51,160,44)', 'rgb(251,154,153)',
            'rgb(227,26,28)', 'rgb(253,191,111)', 'rgb(255,127,0)', 'rgb(202,178,214)', 'rgb(106,61,154)',
            'rgb(255,255,153)', 'rgb(177,89,40)']
rgb_colormap = [re.search(r'rgb\((\d{1,3}),(\d{1,3}),(\d{1,3})\)', color_str).groups() for color_str in rgb_colormap]
hls_colormap = [colorsys.rgb_to_hls(*[float(color)/255 for color in rgb_colors]) for rgb_colors in rgb_colormap]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('cluster_assignment_file', type=Path)
    parser.add_argument('--alpha', type=float, default=0.7, help="Opacity of box fillings")
    args = parser.parse_args()
    cluster_assignment = pd.read_csv(args.cluster_assignment_file)
    cluster_assignment.columns = ['slide_id', 'bounding_box', 'cluster']  # assignment was saved without a label
    num_clusters, num_slides = len(set(cluster_assignment['cluster'])), len(set(cluster_assignment['slide_id']))
    print(f"Making color annotation for {len(cluster_assignment)} objects in {num_slides} slides, divided into {num_clusters} clusters ...")
    if num_clusters > len(hls_colormap):  # sample additional colors uniformly in RGB space
        hls_colormap += np.random.random((num_clusters - len(hls_colormap), 3)).tolist()
    experiment_id = args.cluster_assignment_file.parent.name  # assuming the name of containing dir describes the experiment
    experiment_name = args.cluster_assignment_file.parents[2].name  # similar as above
    layers = tuple(f'Cluster{i}' for i in set(cluster_assignment['cluster']))
    # sort by slide id so only one pointer at a time can be kept for sl
    cluster_assignment = cluster_assignment.set_index(keys='slide_id')
    cluster_assignment = cluster_assignment.sort_index()
    current_slide_id, annotation, num_saved_annotations = '', None, 0
    for i, (slide_id, (box_desc, cluster)) in tqdm(enumerate(cluster_assignment.iterrows()),
                                                   total=len(cluster_assignment)):
        if slide_id != current_slide_id:
            if annotation is not None:
                save_dir = Path(args.data_dir, 'data', 'experiments', experiment_name, 'colorings', experiment_id)
                save_dir.mkdir(exist_ok=True, parents=True)
                annotation.dump_to_json(save_dir)
                num_saved_annotations += 1
                tqdm.write(f"Annotation #{num_saved_annotations} '{slide_id[:12]}...' was saved ({i + 1} objects have been saved into annotations)")
            annotation = AnnotationBuilder(slide_id, experiment_id, layers)
            current_slide_id = slide_id
        x, y, w, h = tuple(int(d) for d in box_desc.split('_'))
        hue, lightness, saturation = hls_colormap[int(cluster)]  # values in [0, 1]
        annotation.add_item(f'Cluster{cluster}', 'rectangle', f'Cluster{cluster}',
                            color={
                                "fill": {
                                    "saturation": saturation,
                                    "lightness": lightness,
                                    "alpha": args.alpha,
                                    "hue": int(hue*360)  # paperjs accepts hue values from 0 to 360
                                },
                                "stroke": {
                                    "saturation": saturation,
                                    "lightness": lightness,
                                    "alpha": 1,
                                    "hue": int(hue*360)
                                }
                            },
                            rectangle={'x': x, 'y': y, 'width': w, 'height': h}
                            )
    print("Done!")



