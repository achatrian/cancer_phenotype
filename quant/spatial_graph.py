from argparse import ArgumentParser
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import networkx as nx


r"""Construct spatial statistics"""


# TODO test
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path, help="Directory storing the WSIs + annotations")
    parser.add_argument('experiment_name', type=str, help="Name of network experiment that produced annotations (annotations are assumed to be stored in subdir with this name)")
    parser.add_argument('clustering_file', type=Path, help="File storing results of clusterings, with index (slide_id, bounding box)")
    parser.add_argument('--distance_threshold', type=float, default=10000)
    args = parser.parse_args()
    feature_dir = args.data_dir/'data'/'features'/args.experiment_name
    clustering_results = pd.read_csv(args.clustering_file)
    # iterate over the slides -- i.e. the files
    for data_path in (feature_dir/'data').iterdir():
        if not data_path.suffix == '.json':
            continue
        with data_path.open('r') as data_file:
            data = json.load(data_file)
        slide_id = data_path.name[5:-5]  # strip 'data_' and '.json'
        slide_component_clustering = clustering_results.filter(like=slide_id)
        if len(slide_component_clustering) == 0:
            warnings.warn(f"No components for slide '{slide_id}'")
        graph = nx.Graph()
        # add nodes to the graph
        for datum in data:
            cluster = slide_component_clustering.filter(like='{}_{}_{}_{}'.format(*datum['bounding_rect']))
            graph.add_node(datum['centroid'], cluster=cluster, bounding_box=datum['bounding_rect'])
        # load distance matrix
        try:
            dist = pd.read_json(feature_dir/'relational'/f'dist_{slide_id}.json')  # ASSUMES DIST WAS SAVED AS A PANDAS DATAFRAME BY analyse.py
        except FileNotFoundError:
            warnings.warn(f"No distance file found for {slide_id}")
            continue
        for centroid0, dist_row in dist.iterrows():
            for centroid1, distance in dist_row.iteritems():
                if distance < args.distance_threshold:
                    # add an edge between two components if they are sufficiently close
                    graph.add_edge(centroid0, centroid1, distance=distance)

        nx.classes.function.info(graph)









