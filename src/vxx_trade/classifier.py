from abc import ABC, abstractmethod
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier


class ClassifierModel(ABC):
    def fit(self, df: pl.DataFrame, features: list[str], target: str) -> None:
        self.model.fit(df.select(features), df.get_column(target))

    def predict(self, df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
        pred = self.model.predict(df.select(features))
        return df.with_columns(pl.Series(pred).alias("prediction"))

    def __repr__(self):
        return f"{self.__class__.__name__}"


class RandomForestClassifierSimple(ClassifierModel):
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
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
        self.model = BayesSearchCV(
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
        self.model = BayesSearchCV(
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
