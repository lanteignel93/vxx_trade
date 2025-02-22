from clustering import *
from vxx_trade.data_generator import generate_data_for_strategy
import numpy as np
import polars as pl
from walkforward import *
from classifier import *
from scaling import *
from winsorization import *

TARGET = "cc_ret"
NUM_BINS = 10
CAT_BINS = 10
MAD_MULTIPLIER = 3


def create_target(df: pl.DataFrame, group_column: str) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col(TARGET).pow(2).sum().alias("group_vol").over(group_column)
        )
        .with_columns(pl.col(TARGET).count().alias("group_count").over(group_column))
        .with_columns(
            pl.lit(16)
            .mul(pl.col(TARGET))
            .truediv((pl.col("group_vol").truediv(pl.col("group_count")).sqrt()))
            .alias("target")
            .over(group_column)
        )
    )


def compute_target(train, test, group_column: str) -> pl.DataFrame:
    train = train.select(["group_vol", "group_count", group_column]).unique()
    test = test.join(train, on=group_column, how="left")
    return test.with_columns(
        pl.lit(16)
        .mul(pl.col(TARGET))
        .truediv((pl.col("group_vol").truediv(pl.col("group_count")).sqrt()))
        .alias("target")
        .over(group_column)
    )


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


if __name__ == "__main__":
    data = generate_data_for_strategy(verbose=False)
    df = data()

    eval_frequency = EvalFrequency.MONTHLY
    start_eval_date = datetime.date(2016, 1, 1)

    wf = WFTrainTestGenerator(eval_frequency, df, start_eval_date)

    features = [
        "vix_cp",
        "vvix_cp",
        "vol_ts",
        "vix_cp_ewma_zscore",
        "vvix_cp_ewma_zscore",
        "vol_ts_ewma_zscore",
    ]

    winsorization = MADWinsorization(MAD_MULTIPLIER)
    scaler = MinMaxScaling()
    cluster = KMeansClustering(CAT_BINS)
    classifier = RandomForestClassifierSimple()
    target_ranker = TargetRanker(NUM_BINS)

    for train, test in wf:

        train = winsorization.fit_transform(train, features)
        test = winsorization.transform(test, features)

        train = scaler.fit_transform(train, features)
        test = scaler.transform(test, features)

        cluster.fit(train, features)
        train = cluster.predict(train, features)
        test = cluster.predict(test, features)

        train = create_target(train, cluster.name)
        test = compute_target(train, test, cluster.name)

        train = target_ranker.fit_transform(train)
        test = target_ranker.transform(test)

        classifier.fit(train, features, "target_rank")
        train = classifier.predict(train, features)
        test = classifier.predict(test, features)

        print(test.select(["date", "target", "target_rank", "prediction"]))
        print(train.select(["date", "target", "target_rank", "prediction"]).describe())
        print(test.select(["date", "target", "target_rank", "prediction"]).describe())

        # print(classification_report(test["target"], preds))
        # print(confusion_matrix(test["target"], preds))
        # print("\n")
        break
