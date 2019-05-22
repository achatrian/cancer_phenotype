from typing import Sequence, Dict
import networkx as nx
import numpy as np

r"""Construct spatial graph"""


class HistoGraph(nx.Graph):

    def __init__(self, slide_id, feature_description=(), incoming_graph_data=None, **attrs):
        super().__init__(incoming_graph_data, **attrs)
        self.slide_id = slide_id
        self.feature_description = feature_description

    def add_feature_nodes_and_edges(self, features: Sequence[Sequence[float]], data: Sequence[Dict], dist: np.array):
        r"""Function to grow graph as features are computed on new instances"""
        assert len(features) == len(data) == dist.shape[0]
        for i, (f, d) in enumerate(zip(features, data)):
            self.add_node(f, coordintates=d['coordinates'], label=d['label'], index=i)
            for j, node_distance in enumerate(dist[i, (i+1):]):
                self.add_edge(f, features[j], weight=node_distance)






