from typing import NamedTuple


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
