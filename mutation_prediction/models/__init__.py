import abc
import math
from typing import Any, Dict, Union

import numpy as np
import sklearn.metrics as metrics
from optuna.trial import Trial
from sklearn.model_selection import KFold
from tqdm import tqdm

import mutation_prediction.data.preprocessing as preprocessing
import mutation_prediction.utils as utils
from mutation_prediction import node
from mutation_prediction.data import Dataset
from mutation_prediction.embeddings import Embedding


class HyperparameterNode(abc.ABC):
    def __init__(self, leaf: bool):
        self.leaf = leaf

    def is_leaf(self) -> bool:
        return self.leaf

    @abc.abstractmethod
    def set_from_trial(self, trial: Trial, params: Dict[str, Any]):
        pass

    @abc.abstractmethod
    def set(self, value: Any):
        pass

    @abc.abstractmethod
    def get(self) -> Any:
        pass


class HyperparameterDict(HyperparameterNode):
    def __init__(self, children: Dict[str, HyperparameterNode]):
        super(HyperparameterDict, self).__init__(False)
        for k, v in children.items():
            if v.is_leaf():
                v.id = k
            else:
                v._set_prefix(k)
        self.children = children
        self.prefix = None

    def __setitem__(self, key: str, value: HyperparameterNode):
        self.children[key] = value
        if value.is_leaf():
            value.id = key
        elif self.prefix is None:
            value._set_prefix(key)
        else:
            value._set_prefix(self.prefix + "_" + key)

    def _set_prefix(self, prefix: str):
        self.prefix = prefix
        for k, v in self.children.items():
            if v.is_leaf():
                v.id = prefix + "_" + k
            else:
                v._set_prefix(prefix + "_" + k)

    def set_from_trial(self, trial: Trial, params: Dict[str, Any]):
        for key, value in self.children.items():
            value.set_from_trial(trial, params[key])

    def set(self, value: Dict[str, Any]):
        for k, v in self.children.items():
            v.set(value[k])

    def get(self) -> Dict[str, Any]:
        values = {}
        for k, v in self.children.items():
            values[k] = v.get()
        return values

    def __getitem__(self, item: str):
        return self.children[item]


class Hyperparameter(HyperparameterNode):
    def __init__(self):
        super(Hyperparameter, self).__init__(True)
        self.id = None
        self.value = None

    def set_from_trial(self, trial: Trial, params: Union[Any, Dict[str, Any]]):
        if isinstance(params, dict):
            self.value = self._suggest(trial, params)
        else:
            self.value = params

    def set(self, value: Any):
        self.value = value

    def get(self) -> Any:
        return self.value

    @abc.abstractmethod
    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        pass


class HyperparameterFloat(Hyperparameter):
    def __init__(self):
        super(HyperparameterFloat, self).__init__()

    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        return trial.suggest_float(self.id, **params)


class HyperparameterInt(Hyperparameter):
    def __init__(self):
        super(HyperparameterInt, self).__init__()

    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        return trial.suggest_int(self.id, **params)


class HyperparameterCategorical(Hyperparameter):
    def __init__(self):
        super(HyperparameterCategorical, self).__init__()

    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        return trial.suggest_categorical(self.id, **params)


class HyperparameterLayersExponential(Hyperparameter):
    def __init__(self, final_size):
        super(HyperparameterLayersExponential, self).__init__()
        self.final_size = final_size

    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        num_hidden = trial.suggest_int(self.id + "_hidden", **params["hidden"])
        if num_hidden == 0:
            return [self.final_size]
        else:
            first_size = trial.suggest_int(self.id + "_first", **params["first"])
            k = (np.log(first_size) - np.log(self.final_size)) / num_hidden
            layers = []
            for i in range(num_hidden + 1):
                layers.append(max(1, int(first_size * np.exp(-k * i))))
            return layers


class HyperparameterCnnExponential(Hyperparameter):
    def __init__(self, input_size):
        super(HyperparameterCnnExponential, self).__init__()
        self.input_size = input_size

    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        num_layers = trial.suggest_int(self.id + "_layers", **params["layers"])
        if num_layers == 0:
            return dict(layers=[], kernel_sizes=[])
        else:
            final_size = trial.suggest_int(self.id + "_final", **params["final"])
            kernel_size = trial.suggest_int(self.id + "_kernel", **params["kernel"])
            k = np.log(final_size / self.input_size) / num_layers
            layers = []
            for i in range(num_layers - 1):
                layers.append(max(1, int(self.input_size * np.exp(k * (i + 1)))))
            layers.append(final_size)
            return dict(layers=layers, kernel_sizes=[kernel_size] * num_layers)


class HyperparameterBool(Hyperparameter):
    def __init__(self):
        super(HyperparameterBool, self).__init__()

    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        return trial.suggest_categorical(self.id, [False, True])


class HyperparameterOptional(HyperparameterNode):
    def __init__(self, child: HyperparameterNode):
        super(HyperparameterOptional, self).__init__(leaf=False)
        self.selected = None
        self.child = child

    def _set_prefix(self, prefix: str):
        self.prefix = prefix
        if self.child.leaf:
            self.child.id = prefix + "_value"
        else:
            self.child._set_prefix(prefix + "_value")

    def set_from_trial(self, trial: Trial, params: Dict[str, Any]):
        if isinstance(params["selected"], dict):
            self.selected = trial.suggest_categorical(self.prefix + "_selected", [False, True])
        else:
            self.selected = params["selected"]
        if self.selected:
            self.child.set_from_trial(trial, params["value"])

    def set(self, value: Any):
        self.selected = value is not None
        if self.selected:
            self.child.set(value)

    def get(self) -> Any:
        if self.selected:
            return self.child.get()
        else:
            return None


class Model(abc.ABC):
    def __init__(self, hyperparams: HyperparameterDict):
        self.hyperparams = hyperparams

    @abc.abstractmethod
    def fit(self, dataset: Dataset, trial: Trial = None):
        pass

    @abc.abstractmethod
    def predict(self, dataset: Dataset) -> np.ndarray:
        pass

    def predict_batched(self, dataset: Dataset, batch_size: int) -> np.ndarray:
        prediction = np.zeros((len(dataset),), dtype=np.float32)
        for i, batch in tqdm(
            enumerate(dataset.in_batches(batch_size)),
            leave=False,
            total=int(math.ceil(len(dataset) / batch_size)),
        ):
            prediction[i * batch_size : i * batch_size + len(batch)] = self.predict(batch)
        return prediction


class FixedEmbeddingModel(Model, abc.ABC):
    def __init__(
        self,
        hyperparams: HyperparameterDict,
        embedding: Embedding,
        normalize_x=False,
        normalize_y=False,
        linearize_x=True,
    ):
        super(FixedEmbeddingModel, self).__init__(hyperparams)
        if hasattr(embedding, "hyperparams"):
            hyperparams["embedding"] = embedding.hyperparams
        self.embedding = embedding
        if normalize_x:
            self.x_normalizer = preprocessing.Normalizer()
        else:
            self.x_normalizer = None
        if normalize_y:
            self.y_normalizer = preprocessing.Normalizer()
        else:
            self.y_normalizer = None
        self.linearize_x = linearize_x

    def fit(self, dataset: Dataset, trial: Trial = None):
        x = self.embedding.embed_update(dataset, trial=trial)
        if self.x_normalizer:
            x = self.x_normalizer.norm_update(x)
        if self.linearize_x and len(x.shape) > 2:
            x = x.reshape((len(dataset), -1))
        y = dataset.get_y()
        if self.y_normalizer:
            y = self.y_normalizer.norm_update(y)
        self._fit(x, y)

    def predict(self, dataset: Dataset) -> np.ndarray:
        x = self.embedding.embed(dataset)
        if self.x_normalizer:
            x = self.x_normalizer.norm(x)
        if self.linearize_x and len(x.shape) > 2:
            x = x.reshape((len(dataset), -1))
        y = self._predict(x)
        if self.y_normalizer:
            y = self.y_normalizer.denorm(y)
        return y

    @abc.abstractmethod
    def _fit(self, x: np.ndarray, y: np.ndarray, trial: Trial = None):
        pass

    @abc.abstractmethod
    def _predict(self, x: np.ndarray) -> np.ndarray:
        pass


class SelfScoringModel(Model):
    def __init__(self, hyperparams: HyperparameterDict):
        super(SelfScoringModel, self).__init__(hyperparams)

    @abc.abstractmethod
    def fit_and_score(self, dataset: Dataset, trial: Trial = None) -> float:
        pass

    def fit(self, dataset: Dataset, trial: Trial = None):
        self.fit_and_score(dataset, trial)


class ModelObjective(abc.ABC):
    def __init__(self, model: Model, params: Dict[str, Any]):
        self.model = model
        self.params = params
        self.trial: Union[None, Trial] = None

    def __call__(self, trial: Trial) -> float:
        self.trial = trial
        trial.set_user_attr("node", node.get_node_name())
        self.model.hyperparams.set_from_trial(trial, self.params)
        return self.run()

    @abc.abstractmethod
    def run(self) -> float:
        pass


class ModelObjectiveIterative(ModelObjective):
    def __init__(self, model: Model, params: Dict[str, Any], score: str):
        super(ModelObjectiveIterative, self).__init__(model, params)
        self.score_main = score
        self.scores = None

    def run(self) -> float:
        self.scores = {
            "rmse": [],
            "r2": [],
        }
        self.evaluate()
        return float(np.median(np.asarray(self.scores[self.score_main])))

    @abc.abstractmethod
    def evaluate(self):
        pass

    def score(self, train: Dataset, validate: Dataset):
        val_prediction = self.model.predict(validate)
        if np.any(np.isnan(val_prediction)):
            val_prediction[:] = np.mean(train.get_y())
        self.scores["rmse"].append(
            metrics.mean_squared_error(validate.get_y(), val_prediction, squared=False)
        )
        self.scores["r2"].append(metrics.r2_score(validate.get_y(), val_prediction))
        self.trial.set_user_attr("scores", self.scores)


class ModelObjectiveCrossValidation(ModelObjectiveIterative):
    def __init__(
        self,
        model: Model,
        params: Dict[str, Any],
        dataset: Dataset,
        splits: int = 5,
        score: str = "rmse",
    ):
        super(ModelObjectiveCrossValidation, self).__init__(model, params, score)
        self.dataset = dataset
        self.splits = splits

    def evaluate(self):
        dataset = self.dataset
        for train_index, _ in KFold(n_splits=self.splits).split(dataset.get_num_mutations()):
            mask = utils.make_mask(len(dataset), *train_index)
            dataset_train, dataset_validate = preprocessing.split_by_mask(dataset, mask)
            dataset_train = preprocessing.shuffle(dataset_train)
            self.model.fit(dataset_train, trial=self.trial)
            self.score(dataset_train, dataset_validate)


class ModelObjectiveFixedValidation(ModelObjectiveIterative):
    def __init__(
        self,
        model: Model,
        params: Dict[str, Any],
        dataset_train: Dataset,
        dataset_val: Dataset,
        iterations: int,
        score: str = "rmse",
    ):
        super(ModelObjectiveFixedValidation, self).__init__(model, params, score)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.iterations = iterations

    def evaluate(self):
        for _ in range(self.iterations):
            dataset_train = preprocessing.shuffle(self.dataset_train)
            self.model.fit(dataset_train, trial=self.trial)
            self.score(self.dataset_train, self.dataset_val)


class ModelObjectiveSelfScoring(ModelObjective):
    def __init__(self, model: SelfScoringModel, params: Dict[str, Any], dataset: Dataset):
        super(ModelObjectiveSelfScoring, self).__init__(model, params)
        self.dataset = dataset

    def run(self) -> float:
        return self.model.fit_and_score(self.dataset, trial=self.trial)
