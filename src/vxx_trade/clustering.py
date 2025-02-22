from abc import ABC, abstractmethod
import polars as pl
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


class ClusteringAlgorithm(ABC):
    def __init__(self, n_clusters: int, name: str):
        self._n_clusters = n_clusters
        self._name = name


    @property
    def n_clusters(self):
        return self._n_clusters

    @property
    def name(self):
        return self._name

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        self.model.fit(df.select(features).to_numpy())

    def predict(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        return df.with_columns(
            pl.Series(self.model.predict(df.select(features).to_numpy()).astype("int32")).alias(self.name)
        )
        
    def __repr__(self):
        return f"{self.name}(n_clusters={self.n_clusters})"


class KMeansClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, random_state: int = 42):
        super().__init__(n_clusters, "KMeansCluster")
        self._model = KMeans(n_clusters=n_clusters, random_state=random_state)

    @property
    def model(self):
        return self._model

class HierarchicalClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters, "HierarchicalCluster")
        self._model = AgglomerativeClustering(n_clusters=n_clusters)

    @property
    def model(self):
        return self._model


class GMMClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, random_state: int = 42):
        super().__init__(n_clusters, "GMMCluster")
        self._model = GaussianMixture(n_components=n_clusters, random_state=random_state)

    @property
    def model(self):
        return self._model

