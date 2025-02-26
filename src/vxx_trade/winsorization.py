from abc import ABC, abstractmethod
from dataclasses import dataclass
import polars as pl
from _utils import CustomEnum


class WinsorizationAlgorithmTypes(CustomEnum):
    MAD = "MADWinsorization"
    Z_SCORE = "ZScoreWinsorization"
    PERCENTILE = "PercentileWinsorization"


class Winsorization(ABC):
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


@dataclass
class WinsorizationParameters:
    winsorization_type: WinsorizationAlgorithmTypes
    kwargs: dict


class WinsorizationFactory:
    def create_winsorization(self, winsorization_type: WinsorizationAlgorithmTypes, *args, **kwargs) -> Winsorization:
        match winsorization_type:
            case WinsorizationAlgorithmTypes.MAD:
                return MADWinsorization(*args, **kwargs)
            case WinsorizationAlgorithmTypes.Z_SCORE:
                return ZScoreWinsorization(*args, **kwargs)
            case WinsorizationAlgorithmTypes.PERCENTILE:
                return PercentileWinsorization(*args, **kwargs)
            case _:
                return ValueError(
                    f"Invalid Winsorization type, choose one of the available options from {' '.join(list(WinsorizationAlgorithmTypes.__members__.keys()))}"
                )

class MADWinsorization(Winsorization):
    def __init__(self, mad_multiplier: int):
        self._mad_multiplier = mad_multiplier
        self._mad: dict = {}
        self._median: dict = {}

    @property
    def mad_multiplier(self) -> int:
        return self._mad_multiplier

    @property
    def mad(self) -> dict:
        return self._mad

    @property
    def median(self) -> dict:
        return self._median

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        for feature in features:
            median = df.get_column(feature).median()
            mad = (df.get_column(feature) - median).abs().median()
            self.mad[feature] = mad
            self.median[feature] = median


    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                pl.when(
                    (pl.col(feature) - self.median[feature]).abs()
                    > self.mad_multiplier * self.mad[feature]
                )
                .then(self.median[feature] + self.mad_multiplier * self.mad[feature])
                .otherwise(
                    pl.when(
                        (pl.col(feature) - self.median[feature]).abs()
                        < -self.mad_multiplier * self.mad[feature],
                    )
                    .then(
                        self.median[feature] - self.mad_multiplier * self.mad[feature]
                    )
                    .otherwise(pl.col(feature))
                )
                .alias(f"{feature}_madwinsorized")
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)


class ZScoreWinsorization(Winsorization):
    def __init__(self, zscore_threshold: int):
        self.zscore_threshold = zscore_threshold
        self.mean: dict = {}
        self.std: dict = {}

    @property
    def zscore_threshold(self):
        return self._zscore_threshold

    @property
    def mean(self) -> dict:
        return self._median

    @property
    def std(self) -> dict:
        return self._std

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        for feature in features:
            self.mean[feature] = df.get_column(feature).mean()
            self.std[feature] = df.get_column(feature).std()

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                pl.when(
                    pl.col(feature)
                    < (self.mean[feature] - self.zscore_threshold * self.std[feature])
                ).then(self.mean[feature] - self.zscore_threshold * self.std[feature])
                .otherwise(
                    pl.when(
                        pl.col(feature)
                        > self.mean[feature]
                        + self.zscore_threshold * self.std[feature]
                ).then(
                        self.mean[feature] + self.zscore_threshold * self.std[feature]
                    ).otherwise(pl.col(feature))
                )
                .alias(f"{feature}_zscorewinsorized")
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)


class PercentileWinsorization(Winsorization):
    def __init__(self, lower_percentile: int, upper_percentile: int):
        self._lower_percentile = lower_percentile
        self._upper_percentile = upper_percentile
        self._lower_bound: dict = {}
        self._upper_bound: dict = {}

    @property
    def lower_percentile(self) -> int:
        return self._lower_percentile

    @property
    def upper_percentile(self) -> int:
        return self._upper_percentile

    @property
    def lower_bound(self) -> dict:
        return self._lower_bound

    @property
    def upper_bound(self) -> dict:
        return self._upper_bound

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        for feature in features:
            self.lower_bound[feature] = df.get_column(feature).quantile(
                self.lower_percentile
            )
            self.upper_bound[feature] = df.get_column(feature).quantile(
                self.upper_percentile
            )

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                pl.when(
                    pl.col(feature) < self.lower_bound[feature]
                ).then(
                    self.lower_bound[feature]
                ).otherwise(
                    pl.when(
                        pl.col(feature) > self.upper_bound[feature]
                    ).then(
                        self.upper_bound[feature],
                    ).otherwise(pl.col(feature))
                )
                .alias(f"{feature}_percentilewinsorized")
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)
