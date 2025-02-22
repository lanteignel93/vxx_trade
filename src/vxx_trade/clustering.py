from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

class ClusteringAlgorithm(ABC):
    def __init__(self, n_clusters: int, name: str):
        self.model = None
        self.n_clusters = n_clusters
        self.name = name

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def predict(self, df):
        pass

    def __repr__(self):
        return f"{self.name}(n_clusters={self.n_clusters})"


class KMeansClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, random_state: int = 42):
        super().__init__(n_clusters, "KMeansCluster")
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, df):
        self.model.fit(df.to_numpy())

    def predict(self, df):
        return self.model.predict(df.to_numpy())


class HierarchicalClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters, "HierarchicalCluster")
        self.model = AgglomerativeClustering(n_clusters=n_clusters)

    def fit(self, df):
        self.model.fit(df.to_numpy())

    def predict(self, df):
        return self.model.fit_predict(df.to_numpy())


class GMMClustering(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, random_state: int = 42):
        super().__init__(n_clusters, "GMMCluster")
        self.model = GaussianMixture(n_components=n_clusters, random_state=random_state)

    def fit(self, df):
        self.model.fit(df.to_numpy())

    def predict(self, df):
        return self.model.predict(df.to_numpy())

