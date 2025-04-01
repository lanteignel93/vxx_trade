from dataclasses import dataclass, asdict, fields
from abc import ABC, abstractmethod
import datetime
import numpy as np
import polars as pl
from vxx_trade.data_generator import  (
    generate_data_for_strategy,
    DataGenerator
)
from clustering import (
    ClusteringParameters,
    ClusteringAlgorithm,
    ClusteringFactory,
    ClusteringAlgorithmTypes
)
from walkforward import (
    WFTrainTestGeneratorParameters,
    WFTrainTestGenerator,
    EvalFrequency
)
from classifier import (
    ClassifierParameters,
    ClassifierModel,
    ClassifierFactory,
    ClassifierAlgorithmTypes
)
from scaling import (
    Scaler,
    ScalingParameters,
    ScalingFactory,
    ScalingAlgorithmTypes
)
from winsorization import (
    Winsorization,
    WinsorizationParameters,
    WinsorizationFactory,
    WinsorizationAlgorithmTypes
)
from target import (
    TargetRankerParameters,
    TargetRanker,
    create_target,
    compute_target
)


@dataclass
class BacktesterConfig:
    target: str = None
    features: list[str] = None
    walkforward_parameters: WFTrainTestGeneratorParameters = None
    winsorization_parameters: WinsorizationParameters = None
    scaling_parameters: ScalingParameters = None
    clustering_parameters: ClusteringParameters = None
    classifier_parameters: ClassifierParameters = None
    target_ranker_parameters: TargetRankerParameters = None

    def __iter__(self):
        return iter(self.__dict__.values())

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, dict) and field.type is not dict:
                setattr(self, field.name, field.type(**value))




class Backtester(BacktesterConfig):
    def __init__(self, config: BacktesterConfig, datagen: DataGenerator | None = None):
        super().__init__(**asdict(config))
        self._data: DataGenerator = None
        self._df: pl.DataFrame = None
        self._get_data(datagen)
        self._wf: WFTrainTestGenerator = self._generate_walkforward()
        self._scaler: Scaler = self._generate_scaler()
        self._winsorization: Winsorization = self._generate_winsorization()
        self._cluster: ClusteringAlgorithm = self._generate_clustering()
        self._classifier: ClassifierModel = self._generate_classifier()
        self._target_ranker: TargetRanker = self._generate_target_ranker()

    @property
    def data(self) -> DataGenerator:
        return self._data

    @data.setter
    def data(self, value: DataGenerator):
        self._data = value

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def wf(self) -> WFTrainTestGenerator:
        return self._wf

    @wf.setter
    def wf(self, value: WFTrainTestGenerator):
        self._wf = value

    @property
    def scaler(self) -> Scaler:
        return self._scaler

    @scaler.setter
    def scaler(self, value: Scaler):
        self._scaler = value

    @property
    def winsorization(self) -> Winsorization:
        return self._winsorization

    @winsorization.setter
    def winsorization(self, value: Winsorization):
        self._winsorization = value

    @property
    def cluster(self) -> ClusteringAlgorithm:
        return self._cluster

    @cluster.setter
    def cluster(self, value: ClusteringAlgorithm):
        self._cluster = value

    @property
    def classifier(self) -> ClassifierModel:
        return self._classifier

    @classifier.setter
    def classifier(self, value: ClassifierModel):
        self._classifier = value

    @property
    def target_ranker(self) -> TargetRanker:
        return self._target_ranker

    @target_ranker.setter
    def target_ranker(self, value: TargetRanker):
        self._target_ranker = value

    def run(self):
        for train, test in self.wf:

            train = self.winsorization.fit_transform(train, self.features)
            test = self.winsorization.transform(test, self.features)

            train = self.scaler.fit_transform(train, self.features)
            test = self.scaler.transform(test, self.features)

            self.cluster.fit(train, self.features)
            train = self.cluster.predict(train, self.features)
            test = self.cluster.predict(test, self.features)

            train = create_target(train, self.cluster.name, self.target)
            test = compute_target(train, test, self.cluster.name, self.target)

            train = self.target_ranker.fit_transform(train)
            test = self.target_ranker.transform(test)

            self.classifier.fit(train, self.features, "target_rank")
            train = self.classifier.predict(train, self.features)
            test = self.classifier.predict(test, self.features)

            print(test.select(["date", "target", "target_rank", "prediction"]))
            print(
                train.select(["date", "target", "target_rank", "prediction"]).describe()
            )
            print(
                test.select(["date", "target", "target_rank", "prediction"]).describe()
            )

            # print(classification_report(test["target"], preds))
            # print(confusion_matrix(test["target"], preds))
            # print("\n")
            break

    def _get_data(self, datagen: DataGenerator | None = None):
        if not datagen:
            self.data = generate_data_for_strategy(verbose=False)
        else:
            self.data = datagen
        self.df = self.data()

    def _generate_walkforward(self) -> WFTrainTestGenerator:
        return WFTrainTestGenerator(df=self.df, **asdict(self.walkforward_parameters))

    def _generate_scaler(self) -> Scaler:
        scaling_factory = ScalingFactory()
        return scaling_factory.create_scaling(
            self.scaling_parameters.scaling_type, **self.scaling_parameters.kwargs
        )

    def _generate_winsorization(self) -> Winsorization:
        winsorization_factory = WinsorizationFactory()
        return winsorization_factory.create_winsorization(
            self.winsorization_parameters.winsorization_type,
            **self.winsorization_parameters.kwargs,
        )

    def _generate_clustering(self) -> ClusteringAlgorithm:
        clustering_factory = ClusteringFactory()
        return clustering_factory.create_clustering(
            self.clustering_parameters.clustering_type,
            **self.clustering_parameters.kwargs,
        )

    def _generate_classifier(self) -> ClassifierModel:
        classifier_factory = ClassifierFactory()
        return classifier_factory.create_classifier(
            self.classifier_parameters.classifier_type,
            **self.classifier_parameters.kwargs,
        )

    def _generate_target_ranker(self) -> TargetRanker:
        return TargetRanker(**self.target_ranker_parameters.kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target})"


class BacktesterBuilder(ABC):
    @abstractmethod
    def build(self) -> Backtester:
        pass

    @abstractmethod
    def set_target(self, target: str):
        pass

    @abstractmethod
    def set_features(self, features: list[str]):
        pass

    @abstractmethod
    def set_walkforward_parameters(
        self, walkforward_parameters: WFTrainTestGeneratorParameters
    ):
        pass

    @abstractmethod
    def set_winsorization_parameters(
        self, winsorization_parameters: WinsorizationParameters
    ):
        pass

    @abstractmethod
    def set_scaling_parameters(self, scaling_parameters: ScalingParameters):
        pass

    @abstractmethod
    def set_clustering_parameters(self, clustering_parameters: ClusteringParameters):
        pass

    @abstractmethod
    def set_classifier_parameters(self, classifier_parameters: ClassifierParameters):
        pass

    @abstractmethod
    def set_target_ranker_parameters(
        self, target_ranker_parameters: TargetRankerParameters
    ):
        pass


class BacktesterDirector:
    def __init__(self, builder: BacktesterBuilder):
        self._builder = builder

    def build(self) -> Backtester:
        self._builder.set_target()
        self._builder.set_features()
        self._builder.set_walkforward_parameters()
        self._builder.set_winsorization_parameters()
        self._builder.set_scaling_parameters()
        self._builder.set_clustering_parameters()
        self._builder.set_classifier_parameters()
        self._builder.set_target_ranker_parameters()

        return self._builder.build()


class BacktesterBuilderExample(BacktesterBuilder):
    def __init__(self):
        self._config = BacktesterConfig()

    def build(self) -> Backtester:
        return Backtester(self._config)

    def set_target(self):
        self._config.target = "cc_ret"

    def set_features(self):
        self._config.features = [
            "vix_cp",
            "vvix_cp",
            "vol_ts",
            "vix_cp_ewma_zscore",
            "vvix_cp_ewma_zscore",
            "vol_ts_ewma_zscore",
        ]

    def set_walkforward_parameters(self):
        self._config.walkforward_parameters = WFTrainTestGeneratorParameters(
            eval_frequency=EvalFrequency.MONTHLY,
            start_eval_date=datetime.date(2016, 1, 1),
        )

    def set_winsorization_parameters(self):
        self._config.winsorization_parameters = WinsorizationParameters(
            winsorization_type=WinsorizationAlgorithmTypes.MAD,
            kwargs={"mad_multiplier": 3},
        )

    def set_scaling_parameters(self):
        self._config.scaling_parameters = ScalingParameters(
            scaling_type=ScalingAlgorithmTypes.MIN_MAX, kwargs={}
        )

    def set_clustering_parameters(self):
        self._config.clustering_parameters = ClusteringParameters(
            clustering_type=ClusteringAlgorithmTypes.KMEANS,
            kwargs={"n_clusters": 10, "random_state": 42},
        )

    def set_classifier_parameters(self):
        self._config.classifier_parameters = ClassifierParameters(
            classifier_type=ClassifierAlgorithmTypes.RANDOM_FOREST_SIMPLE,
            kwargs={"random_state": 42},
        )

    def set_target_ranker_parameters(self):
        self._config.target_ranker_parameters = TargetRankerParameters(
            kwargs={"n_bins": 10}
        )


if __name__ == "__main__":
    builder = BacktesterBuilderExample()
    director = BacktesterDirector(builder)
    backtester = director.build()
    backtester.run()
    print(backtester)
