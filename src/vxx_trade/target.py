import numpy as np
import polars as pl

class TargetRanker:
    def __init__(self, n_bins: int = 10, target: str = "target"):
        self._target = target
        self._n_bins = n_bins
        self._bin_edges: np.ndarray = None

    @property
    def target(self):
        return self._target

    @property
    def n_bins(self):
        return self._n_bins

    @property
    def bin_edges(self):
        return self._bin_edges

    @bin_edges.setter
    def bin_edges(self, value: np.ndarray):
        self._bin_edges = value

    def fit(self, df: pl.DataFrame) -> None:
        # Calculate bin edges
        self.bin_edges = np.linspace(
            df.get_column(self.target).min(),
            df.get_column(self.target).max(),
            self.n_bins + 1,
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        # Digitize the feature values into bins
        ranks = np.digitize(
            df.get_column(self.target), bins=self.bin_edges, right=False
        )
        ranks = ranks.clip(1, self.n_bins)
        return df.with_columns(pl.Series(ranks).alias("target_rank"))

    def fit_transform(self, df) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df)


def create_target(df: pl.DataFrame, group_column: str, target: str) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col(target).pow(2).sum().alias("group_vol").over(group_column)
        )
        .with_columns(pl.col(target).count().alias("group_count").over(group_column))
        .with_columns(
            pl.lit(16)
            .mul(pl.col(target))
            .truediv((pl.col("group_vol").truediv(pl.col("group_count")).sqrt()))
            .alias("target")
            .over(group_column)
        )
    )


def compute_target(train, test, group_column: str, target: str) -> pl.DataFrame:
    train = train.select(["group_vol", "group_count", group_column]).unique()
    test = test.join(train, on=group_column, how="left")
    return test.with_columns(
        pl.lit(16)
        .mul(pl.col(target))
        .truediv((pl.col("group_vol").truediv(pl.col("group_count")).sqrt()))
        .alias("target")
        .over(group_column)
    )

