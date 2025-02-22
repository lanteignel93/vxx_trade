from abc import ABC, abstractmethod
import polars as pl

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


class MADWinsorization(Winsorization):
    def __init__(self, mad_multiplier: int):
        self._mad_multiplier = mad_multiplier
        self._mad : dict = {}
        self._median : dict = {}

    @property
    def mad_multiplier(self):
        return self._mad_multiplier

    @property
    def mad(self):
        return self._mad

    @property
    def median(self):
        return self._median


    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        for feature in features:
            median = df.get_column(feature).median()
            mad = (df.get_column(feature) - median).abs().median()
            self.mad[feature] = mad
            self.median[feature] = median
            lower_bound = median - self.mad_multiplier * mad
            upper_bound = median + self.mad_multiplier * mad
            df = df.with_columns(
                pl.when(df.get_column(feature) < lower_bound).then(lower_bound)
                .when(df.get_column(feature) > upper_bound).then(upper_bound)
                .otherwise(df.get_column(feature))
                .alias(feature)
            )
        return df

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                pl.when(
                    (pl.col(feature) - self.median[feature]).abs() > self.mad_multiplier * self.mad[feature]
                ).then(
                    self.median[feature] + self.mad_multiplier * self.mad[feature]
                ).otherwise(
                    pl.when(
                        (pl.col(feature) - self.median[feature]).abs() < -self.mad_multiplier * self.mad[feature],
                    ).then(
                        self.median[feature] - self.mad_multiplier * self.mad[feature]
                    ).otherwise(pl.col(feature))
                ).alias(f"{feature}_madwinsorized")
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)

#TODO: Fix the Transform based on MAD
class ZScoreWinzorization(Winsorization):
    def __init__(self, zscore_threshold: int):
        self.zscore_threshold = zscore_threshold
        self.mean : dict = {}
        self.std : dict = {}

    @property
    def zscore_threshold(self):
        return self._zscore_threshold

    @property
    def mean(self):
        return self._median

    @property
    def std(self):
        return self._std


    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        for feature in features:
            self.mean[feature] = df.get_column(feature).mean()
            self.std[feature] = df.get_column(feature).std()
            lower_bound = self.mean[feature] - self.zscore_threshold * self.std[feature]
            upper_bound = self.mean[feature] + self.zscore_threshold * self.std[feature]
            df = df.with_columns(
                pl.when(df.get_column(feature) < lower_bound).then(lower_bound)
                .when(df.get_column(feature) > upper_bound).then(upper_bound)
                .otherwise(df.get_column(feature))
                .alias(feature)
            )
        return df

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                f"{feature}_zscorewinsorized",
                pl.when(
                    pl.col(feature) < self.mean[feature] - self.zscore_threshold * self.std[feature],
                    self.mean[feature] - self.zscore_threshold * self.std[feature],
                ).otherwise(
                    pl.when(
                        pl.col(feature) > self.mean[feature] + self.zscore_threshold * self.std[feature],
                        self.mean[feature] + self.zscore_threshold * self.std[feature],
                    ).otherwise(pl.col(feature)),
                ),
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)


#TODO: Fix the Transform based on MAD
class PercentileWinsorization(Winsorization):
    def __init__(self, lower_percentile: int, upper_percentile: int):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bound : dict = {}
        self.upper_bound : dict = {}

    def fit(self, df: pl.DataFrame, features: list[str]) -> None:
        for feature in features:
            self.lower_bound[feature] = df.get_column(feature).quantile(self.lower_percentile)
            self.upper_bound[feature] = df.get_column(feature).quantile(self.upper_percentile)
            df = df.with_columns(
                pl.when(df.get_column(feature) < self.lower_bound[feature]).then(self.lower_bound[feature])
                .when(df.get_column(feature) > self.upper_bound[feature]).then(self.upper_bound[feature])
                .otherwise(df.get_column(feature))
                .alias(feature)
            )
        return df

    def transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        for feature in features:
            df = df.with_columns(
                f"{feature}_percentilewinsorized",
                pl.when(
                    pl.col(feature) < self.lower_bound[feature],
                    self.lower_bound[feature],
                ).otherwise(
                    pl.when(
                        pl.col(feature) > self.upper_bound[feature],
                        self.upper_bound[feature],
                    ).otherwise(pl.col(feature)),
                ),
            )
        return df

    def fit_transform(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        self.fit(df, features)
        return self.transform(df, features)
