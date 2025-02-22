from clustering import *
from vxx_trade.data_generator import generate_data_for_strategy
import polars as pl
from walkforward import *
from classifier import *
from scaling import * 
from winsorization import *

TARGET = "cc_ret"

def create_target(df: pl.DataFrame, group_column: str) -> pl.DataFrame:
    return df.with_columns(
        pl.col(TARGET).pow(2).sum().alias("group_vol").over(group_column)
    ).with_columns(
            pl.col(TARGET).count().alias("group_count").over(group_column)
    ).with_columns(
            pl.lit(16).mul(pl.col(TARGET)).truediv(
                (pl.col("group_vol").truediv(pl.col("group_count")).sqrt())
            ).alias("target").over(group_column)
    )


def compute_target(train, test, group_column: str) -> pl.DataFrame:
    train = train.select(["group_vol", "group_count", group_column]).unique()
    test = test.join(train, on=group_column, how="left")
    return test.with_columns(
            pl.lit(16).mul(pl.col(TARGET)).truediv(
                (pl.col("group_vol").truediv(pl.col("group_count")).sqrt())
            ).alias("target").over(group_column)
        )




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

    winsorization = MADWinsorization(3)
    scaler = MinMaxScaling()
    cluster = KMeansClustering(10)
    classifier = RandomForestClassifierSimple()

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
        print(train.shape)
        print(test.shape)

        classifier.fit(train, features, "target")
        test = classifier.predict(test, features)

        print(test.select(["date", "target", "prediction"]))

        # print(classification_report(test["target"], preds))
        # print(confusion_matrix(test["target"], preds))
        # print("\n")
        break
     
