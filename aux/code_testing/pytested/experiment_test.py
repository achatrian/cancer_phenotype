from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GaussianMixture
from quant.experiment import Experiment


def test_experiment():
    e = Experiment('guassians',
                   ('StandardScaler', 'PCA', 'GaussianMixture'),
                   (StandardScaler(), PCA(n_components=2), GaussianMixture(n_components=5)),
                  caching_path='/home/andrea/Documents/Temp')
    e.read_data_from_dir('/mnt/rescomp/projects/prostate-gland-phenotyping/WSI/data/features', max_memory_use=0)
    e.remove_outliers()  # test if code was fixed
    e.run()
