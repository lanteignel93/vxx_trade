from typing import NamedTuple
from enum import Enum


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


class CustomEnum(Enum):
    @classmethod
    def from_str(cls, label: str):
        try:
            return cls[label.upper()]
        except KeyError:
            raise ValueError(f"{label} is not a valid {cls.__name__}")
