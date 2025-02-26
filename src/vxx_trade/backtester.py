from clustering import *
from vxx_trade.data_generator import generate_data_for_strategy
import numpy as np
import polars as pl
from walkforward import *
from classifier import *
from scaling import *
from winsorization import *
from target import *

TARGET = "cc_ret"
NUM_BINS = 10
CAT_BINS = 10
MAD_MULTIPLIER = 3
START_EVAL_DATE = "2016-01-01"



if __name__ == "__main__":
    data = generate_data_for_strategy(verbose=False)
    df = data()

    eval_frequency = EvalFrequency.MONTHLY
    start_eval_date = datetime.datetime.strptime(START_EVAL_DATE, "%Y-%m-%d").date()

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
