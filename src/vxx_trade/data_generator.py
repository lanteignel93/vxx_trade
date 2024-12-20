import copy
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from vxx_trade import DATA_PATH, DATAGEN_PARAMETERS


@dataclass
class DataGeneratorParameters:
    zscore_period: int
    rank_bucket: int
    ewma_com: int


class DataGenerator(DataGeneratorParameters):
    _volatility_columns = ["vix_cp", "vol_ts", "vvix_cp"]

    def __init__(
        self,
        parameters: DataGeneratorParameters,
        df: pl.DataFrame,
        zscore_period: int | None = None,
        rank_bucket: int | None = None,
        ewma_com: int | None = None,
    ):
        super().__init__(**asdict(parameters))
        self._df = df
        if zscore_period is not None:
            self.zscore_period = zscore_period
        if rank_bucket is not None:
            self.rank_bucket = rank_bucket
        if ewma_com is not None:
            self.ewma_com = ewma_com

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

    @df.setter
    def df(self, df: pl.DataFrame):
        self._df = df

    def compute_trading_data(self) -> pl.DataFrame:
        df = self.df
        df = self.compute_vxx_adjusted_price(df=df)
        df = self.compute_vxx_ret(df=df)
        df = self.compute_term_structure_vol(df=df)

        for vol_col in self._volatility_columns:
            df = self.compute_variable_rank(df=df, column=vol_col)
            df = self.compute_spread_ewma_zscore(df=df, column=vol_col)

        self.df = df
        return df

    def compute_vxx_adjusted_price(self, df: pl.DataFrame) -> pl.DataFrame:
        min_adjustment = df.select(pl.min("adjustmentfactor")).to_numpy()[0][0]
        return (
            df.with_columns(pl.col("adjustmentfactor").truediv(min_adjustment))
            .with_columns(
                pl.col("closeprice").mul(pl.col("adjustmentfactor")).alias("adj_close"),
            )
            .with_columns(
                pl.col("openprice").mul(pl.col("adjustmentfactor")).alias("adj_open")
            )
        )

    def compute_vxx_ret(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col("adj_close").log().diff().shift(-1).alias("cc_ret"))
            .with_columns(
                (pl.col("adj_close").log() - pl.col("adj_open").log()).alias("oc_ret")
            )
            .with_columns(
                (pl.col("adj_open").log().shift(-1) - pl.col("adj_close").log()).alias(
                    "co_ret"
                )
            )
        )

    def compute_spread_ewma_zscore(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        df = self.compute_ewma(df=df, column=column)
        df = df.with_columns(
            (pl.col(f"{column}") - pl.col(f"{column}_ewma")).alias(
                f"{column}_ewma_zscore"
            )
        )

        df = df.with_columns(
            pl.col(f"{column}_ewma_zscore")
            / pl.col(f"{column}_ewma_zscore").rolling_std(
                window_size=self.zscore_period
            )
        )

        return df.with_columns(
            pl.col(f"{column}_ewma_zscore")
            .cut(list(np.arange(-2, 2.5, 0.5)))
            .alias(f"{column}_zscore_bucket")
        )

    def compute_variable_rank(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        return df.with_columns(
            (
                (
                    pl.col(column).rank(method="max", descending=False)
                    / df.height
                    * self.rank_bucket
                )
                .ceil()
                .cast(pl.UInt32)
                .alias(f"{column}_rank")
            )
        )

    def compute_ewma(self, df: pl.DataFrame, column: str) -> pl.DataFrame:
        return df.with_columns(
            pl.col(column).ewm_mean(com=self.ewma_com).alias(f"{column}_ewma")
        )

    def compute_term_structure_vol(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("vix_cp").truediv(pl.col("vix3m_cp")).alias("vol_ts")
        )

    def vxx_reverse_split_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(pl.col("adjustmentfactor").diff().abs().alias("adj_diff"))
            .filter(pl.col("adj_diff").gt(0))
            .select("date")
        )


def generate_data_for_strategy(verbose: bool = True) -> DataGenerator:
    df = pl.read_parquet(DATA_PATH / "vxx_spot.parquet")
    data_parameters = DataGeneratorParameters(**DATAGEN_PARAMETERS)

    data_generator = DataGenerator(parameters=data_parameters, df=df)
    data_generator.compute_trading_data()
    df = data_generator()
    if verbose:
        print(data_generator)
        print(df.tail())
        print(df.columns)

    return data_generator


if __name__ == "__main__":
    generate_data_for_strategy()
