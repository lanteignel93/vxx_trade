import datetime
from enum import Enum
from vxx_trade.data_generator import generate_data_for_strategy
import polars as pl
from dataclasses import dataclass

class EvalFrequency(Enum):
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUALLY = "SEMI_ANNUALLY"
    ANNUALLY = "ANNUALLY"


@dataclass
class WFTrainTestGeneratorParameters:
    eval_frequency: EvalFrequency
    start_eval_date: datetime.date


class WFTrainTestGenerator:
    def __init__(
        self,
        df: pl.DataFrame,
        eval_frequency: EvalFrequency,
        start_eval_date: datetime.date,
    ):
        self.eval_frequency = eval_frequency
        self.days_skipped = None
        self.handle_eval_frequency()
        self.df = df
        self.start_eval_date = start_eval_date
        self.end_eval_date = (
            start_eval_date.replace(day=1) + datetime.timedelta(days=self.days_skipped)
        ).replace(day=1)
        self.end_date = self.df.select("date").max().to_numpy()[0][0].tolist()
        self.stop_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_eval_date > self.end_date or self.stop_flag:
            raise StopIteration
        else:
            train = self.df.filter(pl.col("date") < self.start_eval_date)
            test = self.df.filter(
                (pl.col("date") >= self.start_eval_date)
                & (pl.col("date") < self.end_eval_date)
            )
            prev = self.end_eval_date
            self.end_eval_date = min(
                (
                    self.end_eval_date.replace(day=1)
                    + datetime.timedelta(days=self.days_skipped)
                ).replace(day=1),
                self.end_date,
            )
            self.start_eval_date = min(
                (
                    self.start_eval_date.replace(day=1)
                    + datetime.timedelta(days=self.days_skipped)
                ).replace(day=1),
                self.end_date,
            )
            if self.end_eval_date == prev:
                self.stop_flag = True
            return train.drop_nulls(), test.drop_nulls()

    def handle_eval_frequency(self):
        if self.eval_frequency == EvalFrequency.MONTHLY:
            self.days_skipped = 32
        elif self.eval_frequency == EvalFrequency.QUARTERLY:
            self.days_skipped = 96
        elif self.eval_frequency == EvalFrequency.SEMI_ANNUALLY:
            self.days_skipped = 192
        elif self.eval_frequency == EvalFrequency.ANNUALLY:
            self.days_skipped = 370
        else:
            raise ValueError("Invalid EvalFrequency")


if __name__ == "__main__":
    datagen = generate_data_for_strategy(verbose=False)
    df = datagen()


def test_wf_generator_dates(eval_frequency: EvalFrequency):
    train_test_gen = WFTrainTestGenerator(eval_frequency, df, datetime.date(2016, 1, 1))
    for train, test in train_test_gen:
        print("Train Last Date", train.select("date").max().to_numpy()[0][0].tolist())
        print("Test First Date", test.select("date").min().to_numpy()[0][0].tolist())
        print("Test Last Date", test.select("date").max().to_numpy()[0][0].tolist())
