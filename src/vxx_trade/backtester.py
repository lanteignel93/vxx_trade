from collections.abc import Generator
from abc import ABC, abstractmethod
import datetime
from enum import Enum
from vxx_trade.data_generator import generate_data_for_strategy
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier
import polars as pl


START_EVAL_DATE = datetime.date(2016, 1, 1)
RANDOM_STATE = 42


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
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters, "KMeansCluster")
        self.model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)

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
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters, "GMMCluster")
        self.model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)

    def fit(self, df):
        self.model.fit(df.to_numpy())

    def predict(self, df):
        return self.model.predict(df.to_numpy())


class EvalFrequency(Enum):
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUALLY = "SEMI_ANNUALLY"
    ANNUALLY = "ANNUALLY"


class ClassifierModel(ABC):
    @abstractmethod
    def fit(self, train, target):
        pass

    @abstractmethod
    def predict(self, test):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"


class RandomForestClassifierSimple(ClassifierModel):
    def __init__(self, n_estimators: int = 100, random_state: int = RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )

    def fit(self, train, target) -> None:
        self.model.fit(train, target)

    def predict(self, test):
        return self.model.predict(test)

    def __repr__(self) -> str:
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, random_state={self.random_state})"


class RandomForestClassifierCV(ClassifierModel):
    def __init__(
        self,
        cv: int = 5,
        random_state: int = RANDOM_STATE,
        n_iter: int = 25,
        scoring: str = "accuracy",
        verbose: bool = True,
    ):
        self.params = {
            "n_estimators": Integer(50, 1000),
            "max_depth": Integer(3, 10),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Real(1e-2, 1e-1, prior="log-uniform"),
            "criterion": Categorical(["gini", "entropy", "log_loss"]),
            "max_features": Categorical(["log2", "sqrt"]),
        }
        self.cv = cv
        self.random_state = random_state
        self.n_iter = n_iter
        self.scoring = scoring
        self.verbose = verbose
        self.model = BayesSearchCV(
            RandomForestClassifier(),
            search_spaces=self.params,
            cv=self.cv,
            n_iter=self.n_iter,
            scoring=self.scoring,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def fit(self, train, target) -> None:
        self.model.fit(train, target)

    def predict(self, test):
        return self.model.predict(test)


class XGBClassifierCV(ClassifierModel):
    def __init__(
        self,
        cv: int = 5,
        random_state: int = RANDOM_STATE,
        n_iter: int = 25,
        scoring: str = "accuracy",
        verbose: bool = True,
    ):
        self.params = {
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "n_estimators": Integer(50, 1000),
            "max_depth": Integer(3, 10),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
        }
        self.cv = cv
        self.random_state = random_state
        self.n_iter = n_iter
        self.scoring = scoring
        self.verbose = verbose
        self.model = BayesSearchCV(
            XGBClassifier(),
            search_spaces=self.params,
            cv=self.cv,
            n_iter=self.n_iter,
            scoring=self.scoring,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def fit(self, train, target) -> None:
        self.model.fit(train, target)

    def predict(self, test):
        return self.model.predict(test)


class WFTrainTestGenerator:
    def __init__(
        self,
        eval_frequency: EvalFrequency,
        df: pl.DataFrame,
        start_eval_date: datetime.datetime,
    ):
        self.eval_frequency = eval_frequency
        self.days_skipped = None
        self.handle_eval_frequency()
        self.df = df
        self.start_eval_date = start_eval_date
        self.end_eval_date = (
            start_eval_date.replace(day=1) + datetime.timedelta(days=self.days_skipped)
        ).replace(day=1)
        self.end_date = self.df.select("date").max().to_numpy()[0][0].tolist()
        self.stop_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_eval_date > self.end_date or self.stop_flag:
            raise StopIteration
        else:
            train = self.df.filter(pl.col("date") < self.start_eval_date)
            test = self.df.filter(
                (pl.col("date") >= self.start_eval_date)
                & (pl.col("date") < self.end_eval_date)
            )
            prev = self.end_eval_date
            self.end_eval_date = min(
                (
                    self.end_eval_date.replace(day=1)
                    + datetime.timedelta(days=self.days_skipped)
                ).replace(day=1),
                self.end_date,
            )
            self.start_eval_date = min(
                (
                    self.start_eval_date.replace(day=1)
                    + datetime.timedelta(days=self.days_skipped)
                ).replace(day=1),
                self.end_date,
            )
            if self.end_eval_date == prev:
                self.stop_flag = True
            return train, test

    def handle_eval_frequency(self):
        if self.eval_frequency == EvalFrequency.MONTHLY:
            self.days_skipped = 32
        elif self.eval_frequency == EvalFrequency.QUARTERLY:
            self.days_skipped = 96
        elif self.eval_frequency == EvalFrequency.SEMI_ANNUALLY:
            self.days_skipped = 192
        elif self.eval_frequency == EvalFrequency.ANNUALLY:
            self.days_skipped = 370
        else:
            raise ValueError("Invalid EvalFrequency")


if __name__ == "__main__":
    datagen = generate_data_for_strategy(verbose=False)
    df = datagen()


def test_wf_generator_dates(eval_frequency: EvalFrequency):
    train_test_gen = WFTrainTestGenerator(eval_frequency, df, datetime.date(2016, 1, 1))
    for train, test in train_test_gen:
        print("Train Last Date", train.select("date").max().to_numpy()[0][0].tolist())
        print("Test First Date", test.select("date").min().to_numpy()[0][0].tolist())
        print("Test Last Date", test.select("date").max().to_numpy()[0][0].tolist())
