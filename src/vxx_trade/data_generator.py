import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

DATA_PATH = Path(__file__).parent.resolve() / "data"


@dataclass
class DataGeneratorParameters:
    zscore_period: int
    rank_bucket: int
    ewma_com: int


class DataGenerator(DataGeneratorParameters):
    def __init__(self, parameters: DataGeneratorParameters, df: pl.DataFrame):
        super().__init__(**asdict(parameters))
        self._df = df

    def __call__(self) -> pl.DataFrame:
        return self.df

    def __repr__(self):
        return "<Python VXX Trade %s>" % self.__class__.__name__

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def df(self):
        return self._df


def main():
    df = pl.read_parquet(DATA_PATH / "vxx_spot.parquet")
    df = compute_vxx_adjusted_price(df)
    df = compute_vxx_ret(df)
    df = compute_term_structure_vol(df)
    print(df.select("vol_term_structure").describe())


def compute_vix_spread_ewma_zscore(df: pl.DataFrame, period: int = 21) -> pl.DataFrame:
    df = df.with_columns(
        vix_spread_zscore=(
            pl.col("vix_cp") - pl.col("vix_cp").rolling_mean(window_size=period)
        )
        / pl.col("vix_cp").rolling_std(window_size=period)
    )

    return df.with_columns(
        pl.col("vix_spread_zscore")
        .cut(np.arange(-2, 2.5, 0.5))
        .alias("vix_zscore_bucket")
    )


def compute_variable_rank(
    df: pl.DataFrame, column: str, rank: int = 10
) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col(column).rank(method="max", descending=True) / df.height * rank)
            .ceil()
            .cast(pl.UInt32)
            .alias(f"{column}_rank")
        )
    )


def compute_ewma(df: pl.DataFrame, column: str, com: int) -> pl.DataFrame:
    return df.with_columns(pl.col(column).ewm_mean(com=com).alias(f"{column}_ewma"))


def compute_term_structure_vol(df: pl.DataFrame):
    return df.with_columns(
        pl.col("vix_cp").truediv(pl.col("vixm_cp")).alias("vol_term_structure")
    )


def compute_vxx_adjusted_price(df: pl.DataFrame) -> pl.DataFrame:
    min_adjustment = df.select(pl.min("adjustmentfactor")).to_numpy()[0][0]
    return df.with_columns(
        pl.col("adjustmentfactor").truediv(min_adjustment)
    ).with_columns(
        pl.col("closeprice").mul(pl.col("adjustmentfactor")).alias("adj_price"),
    )


def vxx_reverse_split_dates(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("adjustmentfactor").diff().abs().alias("adj_diff"))
        .filter(pl.col("adj_diff").gt(0))
        .select("date")
    )


def compute_vxx_ret(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("adj_price").log().diff().alias("vxx_log_ret"))


if __name__ == "__main__":
    main()
