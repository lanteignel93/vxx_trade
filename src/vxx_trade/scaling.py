from abc import ABC, abstractmethod
import polars as pl


class Scaling(ABC):

    @abstractmethod
    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        pass

    @abstractmethod
    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        pass


class MinMaxScaling(Scaling):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        self.min = df.select(features).min()
        self.max = df.select(features).max()

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_column(
                f"{feature}_minmax",
                (
                    (pl.col(feature) - self.min[feature])
                    / (self.max[feature] - self.min[feature])
                ).clip(lower=0, upper=1),
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)


class ZScoreScaling(Scaling):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        self.mean = df.select(features).mean()
        self.std = df.select(features).std()

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_column(
                f"{feature}_zscore",
                ((pl.col(feature) - self.mean[feature]) / self.std[feature]),
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)
