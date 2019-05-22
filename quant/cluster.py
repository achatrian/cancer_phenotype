from sklearn import cluster
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
import pandas as pd


def preprocess_featues(x):
    pipeline = Pipeline(steps=[
        ('scale', StandardScaler()),
        ('dim_reduce', PCA)
    ])
    return x.apply(pipeline.fit_transform, raw=True)  # raw=True ensures function is applied to full data ndarray


def kmeans_cluster(x):
    kmeans = cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001)
    labels = kmeans.fit_predict(x.to_numpy())
