from clustering import *
from vxx_trade.data_generator import generate_data_for_strategy
import polars as pl
from walkforward import *
from classifier import *
from scaling import * 
from winsorization import *


if __name__ == "__main__":
    data = generate_data_for_strategy()
    df = data()

    eval_frequency = EvalFrequency.MONTHLY
    start_eval_date = datetime.datetime(2016, 1, 1)

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

        winsorization.fit_transform(train, features)
        winsorization.transform(test, features)

        scaler.fit_transform(train, features)
        scaler.transform(test, features)

        cluster.fit(train, features)
        cluster.transform(test, features)

        classifier.fit(train[features], train["target"])
        preds = classifier.predict(test[features])

        print(classification_report(test["target"], preds))
        print(confusion_matrix(test["target"], preds))
        print("\n")
        break
     
