from __future__ import annotations
from abc import ABC, abstractmethod 
from _utils import CustomEnum, FeatureUpdate, update_dataframe, update_feature_list
from typing import NewType, Union
from dataclasses import dataclass 
import polars as pl
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import Isomap

DimensionReductionType = NewType(
    "DimensionReductionType", Union[
        PCA,
        KernelPCA,
        SparsePCA,
        LinearDiscriminantAnalysis,
        PLSRegression,
        Isomap
    ]
)


class DimensionReductionAlgorithmTypes(CustomEnum):
    PCA = "PCA"
    KERNEL_PCA = "KERNEL_PCA"
    SPARSE_PCA = "SPARSE_PCA"
    LDA = "LDA"
    QDA = "QDA"
    PLS = "PLS"
    ISOMAP = "ISOMAP"


@dataclass 
class DimensionReductionParameters:
    dimension_reduction_technique: DimensionReductionAlgorithmTypes
    kwargs: dict 


class DimensionReduction(ABC):
    def __init__(self, *args, **kwargs):
        self._model = None 

    @property
    def model(self) -> DimensionReductionType:
        return self._model 

    @model.setter
    def model(self, model: DimensionReductionType):
        self._model = model 
    
    @property
    @abstractmethod
    def type_(self) -> DimensionReductionAlgorithmTypes:
        pass
        # return self._type_

    # @type_.setter 
    # def type_(self, type_: DimensionReductionAlgorithmTypes):
        # self._type_ = type_

    def fit(self, df: pl.DataFrame, features: list[str], *args, **kwargs) -> None: 
        X_np = df.select(features).to_numpy()

        if 'target' in kwargs:
            y_np = df.get_column(kwargs['target']).to_numpy()
            self.model.fit(X_np, y_np)

        else:
            self.model.fit(X_np)

    def predict(self, df: pl.DataFrame, features: list[str], output_suffix: str,  n_output: int = 1) -> tuple[pl.DataFrame, FeatureUpdate]:
        X_np = df.select(features).to_numpy()
        output_names = [f'{output_suffix}_{self._type_.name.lower()}_d{i+1}' for i in range(n_output)]

        transform_np = self._model.transform(X_np)[:, 0:n_output]

        df = df.with_columns(
            [
                pl.Series(name=output_names[i], values=transform_np[:, i]) for i in range(n_output)
            ]
        )

        feature_update = FeatureUpdate(
            original_features=features,
            new_features=output_names
        )
        return df, feature_update

    def __repr__(self):
        return f"{self.__class__.__name__}"


class LDA(DimensionReduction):
    def __init__(
        self,
        solver: str = 'eigen'
    ):
        self.solver = solver 
        self._type_ = DimensionReductionAlgorithmTypes.LDA
        self._model = LinearDiscriminantAnalysis(solver=self.solver)

    @property
    def type_(self):
        return self._type_

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(solver={self.solver})'


class PCA_(DimensionReduction):
    def __init__(self):
        self._type_ = DimensionReductionAlgorithmTypes.PCA
        self._model = PCA()

    @property
    def type_(self):
        return self._type_


class KernelPCA_(DimensionReduction):
    def __init__(self, kernel:str = "poly", degree: int = 3):
        self.kernel = kernel 
        self.degree = degree
        self._type_ = DimensionReductionAlgorithmTypes.KERNEL_PCA
        self._model = KernelPCA(
            kernel=self.kernel,
            degree = self.degree
        )

    @property
    def type_(self):
        return self._type_

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(kernel={self.kernel}, degree={self.degree})'


class SparsePCA_(DimensionReduction):
    def __init__(self, alpha : float = 1.0, ridge_alpha : float = 0.01, method: str = 'lars'):
        self.alpha = alpha 
        self.ridge_alpha = ridge_alpha
        self.method = method
        self._type_ = DimensionReductionAlgorithmTypes.SPARSE_PCA
        self._model = SparsePCA(
            alpha=self.alpha,
            ridge_alpha=self.ridge_alpha,
            method=self.method
        )

    @property
    def type_(self):
        return self._type_

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, ridge_alpha={self.ridge_alpha}, method={self.method})'


class PLS(DimensionReduction):
    def __init__(self):
        self._type_ = DimensionReductionAlgorithmTypes.PLS
        self._model = PLSRegression(scale=False)

    @property
    def type_(self):
        return self._type_


class ISOMap(DimensionReduction):
    def __init__(self, n_neighbors: int = 5, n_components: int = 2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self._type_ = DimensionReductionAlgorithmTypes.ISOMAP
        self._model = Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components, n_jobs=-1)

    @property
    def type_(self):
        return self._type_


    def fit(self, df: pl.DataFrame, features: list[str], optimize: bool = False) -> None:
        X_np = df.select(features).to_numpy()
        self.n_components = X_np.ndim

        if optimize:
            reconstruction_errors = {}
            for k in range(4,12):
                try:
                    iso = Isomap(
                        n_neighbors=k,
                        n_components=self.n_components,
                        n_jobs = -1
                    )
                    iso.fit(X_np)
                    reconstruction_errors[k] = iso.reconstruction_errors()

                # FIXME: Proper logging errors later
                except Exception as e:
                    reconstruction_errors[k] = float('inf')
                    pass
            self.n_neighbors = min(reconstruction_errors, key=reconstruction_errors.get)

        self._model = Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components, n_jobs=-1)
        self._model.fit(X_np)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(n_neighbors={self.n_neighbors}, n_components={self.n_components})'




if __name__ == "__main__":
    from vxx_trade.data_generator import generate_data_for_strategy
    import time
    datagen = generate_data_for_strategy()
    df = datagen() 

    features = ['vvix_cp', 'vvix_cp_ewma_zscore']

    df = df.drop_nulls().sample(fraction=1)
    df_test = df.tail(-2000)
    train = df.head(2000)
    test = df.tail(-2000)

    model_list = [PCA_(), KernelPCA_(), SparsePCA_(), ISOMap(), PLS(), LDA()]
    for i, model in enumerate(model_list):
        if i < 4:
            model.fit(df=train, features=features, optimize=True)
        else:
            model.fit(df=train, features=features, target='vvix_cp_rank')

        train, _ = model.predict(df=train, features=features, output_suffix="vvixfeatures", n_output=2)
        test, f_update = model.predict(df=test, features=features, output_suffix="vvixfeatures", n_output=2)

        features = update_feature_list(features, f_update)
        train = update_dataframe(df=train, feature_update=f_update)
        test = update_dataframe(df=test, feature_update=f_update)

        print(train.tail())
        print(test.tail())
        print(features)

        time.sleep(5)
