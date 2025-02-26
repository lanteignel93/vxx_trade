from __future__ import annotations
from abc import ABC
from _utils import CustomEnum
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier
from typing import NewType, Union
from dataclasses import dataclass

ClassifierType = NewType("ClassifierType", Union[RandomForestClassifier, XGBClassifier, BayesSearchCV])


class ClassifierAlgorithmTypes(CustomEnum):
    RANDOM_FOREST_SIMPLE = "RANDOM_FOREST_SIMPLE"
    RANDOM_FOREST_CV = "RANDOM_FOREST_CV"
    XGB_CV = "XGB_CV"


@dataclass
class ClassifierParameters:
    classifier_type: ClassifierAlgorithmTypes
    kwargs: dict


class ClassifierModel(ABC):
    def __init__(self, *args, **kwargs):
        self._model = None

    @property
    def model(self) -> ClassifierType:
        return self._model

    @model.setter
    def model(self, model: ClassifierType) -> ClassifierType:
        self._model = model

    def fit(self, df: pl.DataFrame, features: list[str], target: str) -> None:
        self._model.fit(df.select(features), df.get_column(target))

    def predict(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        pred = self._model.predict(df.select(features))
        return df.with_columns(pl.Series(pred).alias("prediction"))

    def __repr__(self):
        return f"{self.__class__.__name__}"


class ClassifierFactory:
    def create_classifier(self, classifier_type: ClassifierAlgorithmTypes, *args, **kwargs) -> ClassifierModel:
        match classifier_type:
            case ClassifierAlgorithmTypes.RANDOM_FOREST_SIMPLE:
                return RandomForestClassifierSimple(*args, **kwargs)
            case ClassifierAlgorithmTypes.RANDOM_FOREST_CV:
                return RandomForestClassifierCV(*args, **kwargs)
            case ClassifierAlgorithmTypes.XGB_CV:
                return XGBClassifierCV(*args, **kwargs)
            case _:
                return ValueError(f"Invalid Classifier type, choose one of the available options from {' '.join(list(ClassifierAlgorithmTypes.__members__.keys()))}")


class RandomForestClassifierSimple(ClassifierModel):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )

    def __repr__(self) -> str:
        return f"RandomForestClassifier(n_estimators={self.n_estimators}, random_state={self.random_state})"


class RandomForestClassifierCV(ClassifierModel):
    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        n_iter: int = 25,
        scoring: str = "accuracy",
        verbose: bool = True,
    ):
        self.params = {
            "n_estimators": Integer(50, 1000),
            "max_depth": Integer(3, 10),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Real(1e-2, 1e-1, prior="log-uniform"),
            "criterion": Categorical(["gini", "entropy", "log_loss"]),
            "max_features": Categorical(["log2", "sqrt"]),
        }
        self.cv = cv
        self.random_state = random_state
        self.n_iter = n_iter
        self.scoring = scoring
        self.verbose = verbose
        self._model = BayesSearchCV(
            RandomForestClassifier(),
            search_spaces=self.params,
            cv=self.cv,
            n_iter=self.n_iter,
            scoring=self.scoring,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def __repr__(self) -> str:
        return f"RandomForestClassifierCV(params={self.params}"


class XGBClassifierCV(ClassifierModel):
    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        n_iter: int = 25,
        scoring: str = "accuracy",
        verbose: bool = True,
    ):
        self.params = {
            "learning_rate": Real(0.01, 1.0, prior="log-uniform"),
            "n_estimators": Integer(50, 1000),
            "max_depth": Integer(3, 10),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
        }
        self.cv = cv
        self.random_state = random_state
        self.n_iter = n_iter
        self.scoring = scoring
        self.verbose = verbose
        self._model = BayesSearchCV(
            XGBClassifier(),
            search_spaces=self.params,
            cv=self.cv,
            n_iter=self.n_iter,
            scoring=self.scoring,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def __repr__(self) -> str:
        return f"XGBClassifierCV(params={self.params}"
