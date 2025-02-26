from _utils import CustomEnum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import polars as pl


class ScalingAlgorithmTypes(CustomEnum):
    MIN_MAX = "MinMaxScaling"
    Z_SCORE = "ZScoreScaling"


@dataclass
class ScalingParameters:
    scaling_type: ScalingAlgorithmTypes
    kwargs: dict


class Scaler(ABC):
    @abstractmethod
    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        pass

    @abstractmethod
    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        pass

    @abstractmethod
    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ScalingFactory:
    def create_scaling(
        self, scaling_type: ScalingAlgorithmTypes, *args, **kwargs
    ) -> Scaler:
        match scaling_type:
            case ScalingAlgorithmTypes.MIN_MAX:
                return MinMaxScaling(*args, **kwargs)
            case ScalingAlgorithmTypes.Z_SCORE:
                return ZScoreScaling(*args, **kwargs)
            case _:
                return ValueError(
                    f"Invalid Scaling type, choose one of the available options from {' '.join(list(ScalingAlgorithmTypes.__members__.keys()))}"
                )


class MinMaxScaling(Scaler):
    def __init__(self):
        self._min: pl.Series = None
        self._max: pl.Series = None

    @property
    def min(self) -> pl.Series:
        return self._min

    @property
    def max(self) -> pl.Series:
        return self._max

    @min.setter
    def min(self, value: pl.Series):
        self._min = value

    @max.setter
    def max(self, value: pl.Series):
        self._max = value

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        self.min = df.select(features).min()
        self.max = df.select(features).max()

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                (
                    (pl.col(feature) - self.min.get_column(feature))
                    / (self.max.get_column(feature) - self.min.get_column(feature))
                )
                .clip(lower_bound=0, upper_bound=1)
                .alias(f"{feature}_minmax")
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)


class ZScoreScaling(Scaler):
    def __init__(self):
        self._mean: pl.Series = None
        self._std: pl.Series = None

    @property
    def mean(self) -> pl.Series:
        return self._mean

    @property
    def std(self) -> pl.Series:
        return self._std

    @mean.setter
    def mean(self, value: pl.Series):
        self._mean = value

    @std.setter
    def std(self, value: pl.Series):
        self._std = value

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        self.mean = df.select(features).mean()
        self.std = df.select(features).std()

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                f"{feature}_zscore",
                (
                    (pl.col(feature) - self.mean.get_column(feature))
                    / self.std.get_column(feature)
                ),
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)
