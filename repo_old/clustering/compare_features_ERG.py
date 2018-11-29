import os, sys
import argparse
import numpy as np
from pathlib import Path
from itertools import product
import multiprocessing as mp

import matplotlib.pyplot as plt
import seaborn as sns


# Comparing of CK5 clustering fitness for different feature extraction methods

seg_dir = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/cluster_seg"
ck5_dir = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/cluster_ck5"
hc_dir = "/Users/andreachatrian/Documents/Repositories/cancer_phenotype/Results/handcrafted"

# Load fitnesses
with open(os.path.join(seg_dir, "query_seg_erg.csv"), 'r') as query_file:
    fitness_seg = np.loadtxt(query_file)

with open(os.path.join(ck5_dir, "query_erg.csv"), 'r') as query_file:
    fitness_ck5 = np.loadtxt(query_file)

with open(os.path.join(hc_dir, "query_hc_erg.csv"), 'r') as query_file:
    fitness_hc = np.loadtxt(query_file)

# Query clusters
chance_levels = np.ones((fitness_seg.shape[0], 2)) * np.array([0.78, 0])  # distribution of labels in dataset
all_query_fits = np.concatenate((fitness_seg, fitness_ck5, fitness_hc, chance_levels), axis=1)
k_range = np.arange(2, 20, 2)
legend_entries = [label + method for label, method in product(["u-net", "inception", "handcrafted", "chance"],
                                                              ["erg + ", "erg - "])]
print(legend_entries)
sns.set(style="darkgrid")
palette = sns.color_palette("hls", 8)
sns.set_palette(palette)
plt.plot(np.tile(k_range[:, np.newaxis], (1, 8)), all_query_fits)
plt.legend(legend_entries, loc='center right', prop=dict(size=12))
plt.title("Cluster query fitness for different neighbourhood sizes", weight="bold").set_fontsize('14')
plt.show(block=True)