from typing import Any, Dict, Union

import numpy as np
import optuna
from optuna import Trial
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR

from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import (
    FixedEmbeddingModel,
    Hyperparameter,
    HyperparameterCategorical,
    HyperparameterDict,
    HyperparameterFloat,
    HyperparameterInt,
)


class PlsRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(PlsRegressor, self).__init__(
            HyperparameterDict(
                {
                    "n_components": HyperparameterInt(),
                }
            ),
            embedding,
        )
        self.model: Union[PLSRegression, None] = None

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        self.model = PLSRegression(n_components=self.hyperparams["n_components"].get())
        self.model.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)[:, 0]


class LinearRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(LinearRegressor, self).__init__(HyperparameterDict({}), embedding)
        self.model: Union[LinearRegression, None] = None

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        self.model = LinearRegression()
        self.model.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class HyperparameterKernel(Hyperparameter):
    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        if isinstance(params["kernel"], list):
            kernel = trial.suggest_categorical(self.id + "_kernel", choices=params["kernel"])
        elif isinstance(params["kernel"], str):
            kernel = params["kernel"]
        else:
            raise ValueError("Invalid kernel!")
        if kernel == "linear":
            return dict(kernel=kernel)
        elif kernel == "polynomial":
            if isinstance(params["degree"], int):
                degree = params["degree"]
            elif isinstance(params["degree"], dict):
                degree = trial.suggest_int(self.id + "_degree", **params["degree"])
            else:
                raise ValueError("Invalid polynomial kernel degree!")
            return dict(
                kernel=kernel,
                degree=degree,
            )
        elif kernel == "rbf":
            if isinstance(params["gamma"], float) or isinstance(params["gamma"], int):
                gamma = float(params["gamma"])
            elif isinstance(params["gamma"], dict):
                gamma = trial.suggest_float(self.id + "_gamma", **params["gamma"])
            else:
                raise ValueError("Invalid RBF kernel gamma value!")
            return dict(
                kernel=kernel,
                gamma=gamma,
            )
        else:
            raise ValueError("Invalid kernel!")


class EpsilonSvr(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(EpsilonSvr, self).__init__(
            HyperparameterDict(
                {
                    "C": HyperparameterFloat(),
                    "epsilon": HyperparameterFloat(),
                    "gamma": HyperparameterFloat(),
                }
            ),
            embedding,
            normalize_y=True,
        )

    def _fit(self, x: np.ndarray, y: np.ndarray, trial: Trial = None):
        if x.size == 0:
            raise optuna.TrialPruned()
        self.model = SVR(
            C=self.hyperparams["C"].get(),
            epsilon=self.hyperparams["epsilon"].get(),
            kernel="rbf",
            gamma=self.hyperparams["gamma"].get(),
            cache_size=500,  # 500 MB
        )
        self.model.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class EpsilonSvrLinear(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(EpsilonSvrLinear, self).__init__(
            HyperparameterDict(
                {
                    "C": HyperparameterFloat(),
                    "epsilon": HyperparameterFloat(),
                    "loss": HyperparameterCategorical(),
                }
            ),
            embedding,
            normalize_y=True,
        )

    def _fit(self, x: np.ndarray, y: np.ndarray, trial: Trial = None):
        if x.size == 0:
            raise optuna.TrialPruned()
        loss = {
            "l1": "epsilon_insensitive",
            "l2": "squared_epsilon_insensitive",
        }[self.hyperparams["loss"].get()]
        self.model = LinearSVR(
            C=self.hyperparams["C"].get(),
            epsilon=self.hyperparams["epsilon"].get(),
            loss=loss,
            max_iter=10000000,
        )
        self.model.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)
