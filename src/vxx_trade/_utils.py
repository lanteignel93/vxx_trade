from typing import NamedTuple
from enum import Enum
import polars as pl


class MatplotlibFigSize(NamedTuple):
    width: int
    height: int


class MatplotlibAxesLimit(NamedTuple):
    min: float | int
    max: float | int


class YearsResearch(NamedTuple):
    start: int
    end: int


class TargetColumn(NamedTuple):
    name: str
    return_type: str

class FeatureUpdate(NamedTuple):
    original_features : list[str]
    new_features: list[str]


def update_dataframe(df: pl.DataFrame, feature_update: FeatureUpdate) -> pl.DataFrame:
    return df.drop(feature_update.original_features)

def update_feature_list(list_features: list[str], feature_update: FeatureUpdate) -> list[str]:
    return list(
        set(list_features)
        .symmetric_difference(
            set(feature_update.original_features)
            .union(set(feature_update.new_features))
        )
    )


class CustomEnum(Enum):
    @classmethod
    def from_str(cls, label: str):
        try:
            return cls[label.upper()]
        except KeyError:
            raise ValueError(f"{label} is not a valid {cls.__name__}")
