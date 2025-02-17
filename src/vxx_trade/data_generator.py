import copy
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from vxx_trade import DATA_PATH, DATAGEN_PARAMETERS


@dataclass
class DataGeneratorParameters:
    """
    Data class to hold parameters for data generation.
    
    Attributes:
        zscore_period (int): Period for z-score calculation.
        rank_bucket (int): Number of buckets for ranking.
        ewma_com (int): Center of mass for EWMA calculation.
    """
    zscore_period: int
    rank_bucket: int
    ewma_com: int


class DataGenerator(DataGeneratorParameters):
    """
    Class to generate trading data for VXX strategy.
    Inherits from DataGeneratorParameters.
    
    Attributes:
        _volatility_columns (list): List of volatility-related column names.
        _df (pl.DataFrame): DataFrame containing the trading data.
    """
    _volatility_columns = ["vix_cp", "vol_ts", "vvix_cp"]

    def __init__(
        self,
        parameters: DataGeneratorParameters,
        df: pl.DataFrame,
        zscore_period: int | None = None,
        rank_bucket: int | None = None,
        ewma_com: int | None = None,
    ):
        """
        Initialize the DataGenerator with parameters and DataFrame.
        
        Args:
            parameters (DataGeneratorParameters): Parameters for data generation.
            df (pl.DataFrame): DataFrame containing the trading data.
            zscore_period (int, optional): Period for z-score calculation.
            rank_bucket (int, optional): Number of buckets for ranking.
            ewma_com (int, optional): Center of mass for EWMA calculation.
        """
        super().__init__(**asdict(parameters))
        self._df = df
        if zscore_period is not None:
            self.zscore_period = zscore_period
        if rank_bucket is not None:
            self.rank_bucket = rank_bucket
        if ewma_com is not None:
            self.ewma_com = ewma_com

    def __call__(self) -> pl.DataFrame:
        """
        Return the DataFrame.
        
        Returns:
            pl.DataFrame: The DataFrame containing the trading data.
        """
        return self.df

    def __repr__(self):
        """
        String representation of the class.
        
        Returns:
            str: String representation of the class.
        """
        return "<Python VXX Trade %s>" % self.__class__.__name__

    def __copy__(self):
        """
        Create a shallow copy of the instance.
        
        Returns:
            DataGenerator: A shallow copy of the instance.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict):
        """
        Create a deep copy of the instance.
        
        Args:
            memo (dict): Memoization dictionary for deep copy.
        
        Returns:
            DataGenerator: A deep copy of the instance.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    @property
    def df(self):
        """
        Getter for the DataFrame.
        
        Returns:
            pl.DataFrame: The DataFrame containing the trading data.
        """
        return self._df

    @df.setter
    def df(self, df: pl.DataFrame):
        """
        Setter for the DataFrame.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
        """
        self._df = df

    def compute_trading_data(self) -> pl.DataFrame:
        """
        Compute various trading data metrics and update the DataFrame.
        
        Returns:
            pl.DataFrame: The updated DataFrame with computed metrics.
        """
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
        """
        Adjust prices based on the adjustment factor.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
        
        Returns:
            pl.DataFrame: The DataFrame with adjusted prices.
        """
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
        """
        Compute returns based on adjusted prices.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
        
        Returns:
            pl.DataFrame: The DataFrame with computed returns.
        """
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
        """
        Compute EWMA z-scores and bucket them.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
            column (str): The column name to compute EWMA z-scores for.
        
        Returns:
            pl.DataFrame: The DataFrame with computed EWMA z-scores and buckets.
        """
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
        """
        Rank variables and bucket them.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
            column (str): The column name to rank.
        
        Returns:
            pl.DataFrame: The DataFrame with ranked variables and buckets.
        """
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
        """
        Compute the exponentially weighted moving average.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
            column (str): The column name to compute EWMA for.
        
        Returns:
            pl.DataFrame: The DataFrame with computed EWMA.
        """
        return df.with_columns(
            pl.col(column).ewm_mean(com=self.ewma_com).alias(f"{column}_ewma")
        )

    def compute_term_structure_vol(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute term structure volatility.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
        
        Returns:
            pl.DataFrame: The DataFrame with computed term structure volatility.
        """
        return df.with_columns(
            pl.col("vix_cp").truediv(pl.col("vix3m_cp")).alias("vol_ts")
        )

    def vxx_reverse_split_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Identify reverse split dates based on adjustment factor changes.
        
        Args:
            df (pl.DataFrame): The DataFrame containing the trading data.
        
        Returns:
            pl.DataFrame: The DataFrame with identified reverse split dates.
        """
        return (
            df.with_columns(pl.col("adjustmentfactor").diff().abs().alias("adj_diff"))
            .filter(pl.col("adj_diff").gt(0))
            .select("date")
        )


def generate_data_for_strategy(verbose: bool = True) -> DataGenerator:
    """
    Generate data for the VXX trading strategy.
    
    Args:
        verbose (bool, optional): If True, print the data generator and DataFrame info.
    
    Returns:
        DataGenerator: The DataGenerator instance with computed trading data.
    """
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
