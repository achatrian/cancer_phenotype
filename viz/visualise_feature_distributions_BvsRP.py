from pathlib import Path
import json
from collections import namedtuple
from contextlib import contextmanager
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
plt.rc('font', size='20')
distribution = namedtuple('distribution', ['id_', 'min', 'max', 'mean', 'std', 'iqr', 'q5', 'q95',
                                           'n', 'num_bins', 'distribution', 'columns'])


r"""Save plots of features histograms for all features in biopsy vs RP specimens"""


@contextmanager
def figure(save_path):
    # Code to acquire resource, e.g.:
    fig = plt.figure()
    try:
        yield fig
    finally:
        fig.savefig(save_path)
        plt.close(fig)


if __name__ == '__main__':
    save_dir = Path('/well/rittscher/projects/ProMPT/data/results/features_comparison')
    save_dir.mkdir(exist_ok=True, parents=True)
    master_list = pd.read_excel('/well/rittscher/projects/ProMPT/cases/ProMPT_master_list.xlsx', sheet_name='cases')
    master_list = master_list.set_index('SpecimenIdentifier')
    with open('/well/rittscher/projects/ProMPT/data/features/combined_mpp1.0_normal_nuclei_circles_good_nuc_distribution_100bins.json', 'r') as distribution_file:
        distributions_ = json.load(distribution_file)
    errors_log = []
    def make_dist(dist_):
        elements = []
        for el in dist_:
            try:
                el = pd.read_json(el, orient='records')
            except Exception as err:
                errors_log.append((type(err), err))
            elements.append(el)
        try:
            new_dist = distribution(*elements)
        except TypeError:
            print(f"Incorrect number of elements: {len(elements)} for {dist_[0]}")
            raise
        return new_dist
    distributions = [make_dist(dist_) for dist_ in distributions_]
    dataset_distribution = distributions[-1]
    distributions = distributions[:-1]
    b_rp_labels = np.array(['b' if master_list.loc[int(d.id_.split('_')[0])]['SpecimenType'] == 'Biopsy' else 'rp' for d in distributions])
    for feature_name in tqdm(distributions[0].distribution.columns):
        feature_dir = save_dir/feature_name
        feature_dir.mkdir(exist_ok=True)
        # check mean histograms for biopsies and RPs
        rp_histograms, biopsy_histograms, b_rp_slide_ids = [], [], {'b': [], 'rp': []}
        for d in distributions:
            case_id = d.id_.split('_')[0]
            if master_list.loc[int(case_id), 'SpecimenType'] == 'Biopsy':
                biopsy_histograms.append(d.distribution[feature_name])
                b_rp_slide_ids['b'].append(d.id_)
            if master_list.loc[int(case_id), 'SpecimenType'] == 'RP':
                rp_histograms.append(d.distribution[feature_name])
                b_rp_slide_ids['rp'].append(d.id_)
        biopsy_histograms, rp_histograms = np.array(biopsy_histograms), np.array(rp_histograms)
        biopsy_densities, rp_densities = (biopsy_histograms.T/biopsy_histograms.sum(axis=1)).T, (rp_histograms.T/rp_histograms.sum(axis=1)).T
        mean_b_density, mean_rp_density = np.mean(biopsy_densities, axis=0), np.mean(rp_densities, axis=0)
        std_b_density, std_rp_density = np.std(biopsy_densities, axis=0), np.std(rp_densities, axis=0)
        with figure(feature_dir/f'{feature_name}_dens_biopsy.png'):
            max_ = np.max(np.concatenate((mean_b_density, mean_rp_density)))
            min_mean = np.min((mean_b_density[mean_b_density != 0].min(), mean_rp_density[mean_rp_density != 0].min()))
            min_std = np.min((std_b_density[std_b_density != 0].min(), std_rp_density[std_rp_density != 0].min()))
            plt.bar(np.arange(len(mean_b_density)), mean_b_density)
            plt.bar(np.arange(len(mean_b_density)), std_b_density - min_std, alpha=0.4)
            plt.title(f'{feature_name} density - Biopsy')
            plt.ylim((0, max_))
        with figure(feature_dir/f'{feature_name}_dens_RP.png'):
            plt.bar(np.arange(len(mean_b_density)), mean_rp_density - min_mean)
            plt.bar(np.arange(len(mean_b_density)), std_rp_density - min_std, alpha=0.4)
            plt.title(f'{feature_name} density - RPs')
            plt.ylim((0, max_))
        with figure(feature_dir/f'{feature_name}_log_dens_Biopsy.png'):
            min_mean = np.min((mean_b_density[mean_b_density != 0].min(), mean_rp_density[mean_rp_density != 0].min()))
            min_std = np.min((std_b_density[std_b_density != 0].min(), std_rp_density[std_rp_density != 0].min()))
            all_ = np.concatenate((np.log10(mean_b_density), np.log10(mean_rp_density))) - np.log10(min_mean)
            plt.bar(np.arange(len(mean_b_density)), np.log10(mean_b_density) - np.log10(min_mean))
            plt.bar(np.arange(len(mean_b_density)), np.log10(std_b_density) - np.log10(min_std), alpha=0.4)
            plt.title(f'{feature_name} log density - Biopsy')
            plt.ylim(0, all_[all_ != -np.inf].max())
        with figure(feature_dir/f'{feature_name}_log_dens_RP.png'):
            plt.bar(np.arange(len(mean_b_density)), np.log10(mean_rp_density) - np.log10(min_mean))
            plt.bar(np.arange(len(mean_b_density)), np.log10(std_rp_density) - np.log10(min_std), alpha=0.4)
            plt.title(f'{feature_name} log density - RPs')
            plt.ylim(0, all_[all_ != -np.inf].max())
        with figure(feature_dir/f'{feature_name}_dens_diff.png'):
            diff_means = np.abs((mean_b_density - mean_rp_density))
            plt.bar(np.arange(len(mean_b_density)), diff_means - diff_means[diff_means != -np.inf].min())
            plt.title(f'{feature_name} densities difference - Biopsies vs RPs')
        with figure(feature_dir/f'{feature_name}_log_dens_diff.png'):
            diff_means = np.log10(diff_means)
            plt.bar(np.arange(len(mean_b_density)), diff_means - diff_means[diff_means != -np.inf].min())
            plt.title(f'{feature_name} densities difference - Biopsies vs RPs')
        # save PCA
        distX = np.array([d.distribution[feature_name].tolist() for d in distributions])
        distXpca = PCA(2, whiten=True).fit_transform(distX)
        with figure(feature_dir/f'{feature_name}_pca.png'):
            y = plt.scatter(distXpca[b_rp_labels == 'b'][:, 0], distXpca[b_rp_labels == 'b'][:, 1], c='y', alpha=0.5)
            c = plt.scatter(distXpca[b_rp_labels == 'rp'][:, 0], distXpca[b_rp_labels == 'rp'][:, 1], c='c', alpha=0.5)
            plt.legend((y, c), ('biopsies', 'radical prostatectomies'))
    print("Done!")



