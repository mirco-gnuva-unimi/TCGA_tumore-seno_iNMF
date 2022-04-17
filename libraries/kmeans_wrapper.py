from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


def kmeans_clusters(dataset: pd.DataFrame, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataset.to_numpy())
    return list(kmeans.labels_)


def optimize_n_clusters(dataset: pd.DataFrame, n_max: int) -> int:
    n_min = 2
    silhouettes = [silhouette_score(dataset.to_numpy(), kmeans_clusters(dataset, n_clusters)) for n_clusters in
                   range(n_min, n_max + 1)]

    return np.argmax(silhouettes) + n_min
