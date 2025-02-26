from __future__ import annotations
from abc import ABC
from _utils import CustomEnum
import polars as pl
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from typing import NewType, Union
from dataclasses import dataclass

ClusterType = NewType(
    "ClusterType", Union[KMeans, AgglomerativeClustering, GaussianMixture]
)


class ClusteringAlgorithmTypes(CustomEnum):
    KMEANS = "KMeansClustering"
    HIERARCHICAL = "HierarchicalClustering"
    GMM = "GMMClustering"


@dataclass
class ClusteringParameters:
    clustering_type: ClusteringAlgorithmTypes
    kwargs: dict


class ClusteringAlgorithm(ABC):
    def __init__(self, n_clusters: int, name: str):
        self._n_clusters = n_clusters
        self._name = name
        self._model: ClusterType = None

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> ClusterType:
        return self._model

    @model.setter
    def model(self, model: ClusterType) -> ClusterType:
        self._model = model

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        self.model.fit(df.select(features).to_numpy())

    def predict(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        return df.with_columns(
            pl.Series(
                self.model.predict(df.select(features).to_numpy()).astype("int32")
            ).alias(self.name)
        )

    def __repr__(self):
        return f"{self.name}(n_clusters={self.n_clusters})"


class ClusteringFactory:
    def create_clustering(
        self, clustering_type: ClusteringAlgorithmTypes, *args, **kwargs
    ) -> ClusteringAlgorithm:
        match clustering_type:
            case ClusteringAlgorithmTypes.KMEANS:
                return KMeansClustering(*args, **kwargs)
            case ClusteringAlgorithmTypes.HIERARCHICAL:
                return HierarchicalClustering(*args, **kwargs)
            case ClusteringAlgorithmTypes.GMM:
                return GMMClustering(*args, **kwargs)
            case _:
                return ValueError(
                    f"Wrong clustering type, choose one of the available options between {' '.join(list(ClusteringAlgorithmTypes.__members__.keys()))}"
                )


class KMeansClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, random_state: int = 42):
        super().__init__(n_clusters, "KMeansCluster")
        self._model = KMeans(n_clusters=n_clusters, random_state=random_state)


class HierarchicalClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters, "HierarchicalCluster")
        self._model = AgglomerativeClustering(n_clusters=n_clusters)


class GMMClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, random_state: int = 42):
        super().__init__(n_clusters, "GMMCluster")
        self._model = GaussianMixture(
            n_components=n_clusters, random_state=random_state
        )
