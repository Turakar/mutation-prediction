import multiprocessing
import time
from typing import Union

import numpy as np
import optuna
from optuna import Trial
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

import mutation_prediction.node as node
from mutation_prediction.data import Dataset
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import (
    HyperparameterDict,
    HyperparameterInt,
    SelfScoringModel,
)


class SelfOptimizedSvm(SelfScoringModel):
    def __init__(self, embedding: Embedding):
        hyperparams = HyperparameterDict(
            {
                "trials": HyperparameterInt(),
                "splits": HyperparameterInt(),
            }
        )
        if hasattr(embedding, "hyperparams"):
            hyperparams["embedding"] = embedding.hyperparams
        super(SelfOptimizedSvm, self).__init__(hyperparams)
        self.embedding = embedding
        self.model: Union[None, SVR] = None

    def fit_and_score(self, dataset: Dataset, trial: Trial = None) -> float:
        # we distinguish runs with a known sub-optimization result (frozen trial) and without
        writable_trial = trial is not None and not isinstance(trial, optuna.trial.FrozenTrial)
        frozen_trial = trial is not None and isinstance(trial, optuna.trial.FrozenTrial)

        # embed data
        prior = time.time()
        x = self.embedding.embed_update(dataset, trial=trial)
        if len(x.shape) > 2:
            x = x.reshape((len(dataset), -1))
        if writable_trial:
            trial.set_user_attr("time_embedding", time.time() - prior)

        if not frozen_trial:
            # create sub study for optimizing C and gamma
            storage_url = node.get_redis_url()
            study_name = "sub-study"
            storage = optuna.storages.RedisStorage(storage_url)
            sub_study = optuna.create_study(
                storage=storage, study_name=study_name, direction="minimize"
            )

            # start multiprocess optimization
            prior = time.time()
            num_processes = node.get_num_cpus()
            processes = []
            for i in range(num_processes):
                p = multiprocessing.Process(
                    target=_self_optimized_svm_worker,
                    args=(
                        storage_url,
                        study_name,
                        x,
                        dataset.get_y(),
                        self.hyperparams["trials"].get(),
                        self.hyperparams["splits"].get(),
                    ),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            if writable_trial:
                trial.set_user_attr("time_optimize", time.time() - prior)

            # retrieve best params
            best_c = sub_study.best_params["C"]
            best_gamma = sub_study.best_params["gamma"]
            value = sub_study.best_value
            if writable_trial:
                trial.set_user_attr("scores", sub_study.best_trial.user_attrs["scores"])
                value_history = [trial.value for trial in sub_study.trials]
                trial.set_user_attr("sub_optimization_history", value_history)
                trial.set_user_attr("best_params", sub_study.best_params)

            # delete sub study
            sub_study_id = storage.get_study_id_from_name(study_name)
            storage.delete_study(sub_study_id)

        else:
            # retrieve existing values
            best_c = trial.user_attrs["best_params"]["C"]
            best_gamma = trial.user_attrs["best_params"]["gamma"]
            value = float("nan")

        # train final model
        prior = time.time()
        self.model = SVR(C=best_c, gamma=best_gamma)
        self.model.fit(x, dataset.get_y())
        if writable_trial:
            trial.set_user_attr("time_train_final", time.time() - prior)

        return value

    def predict(self, dataset: Dataset) -> np.ndarray:
        x = self.embedding.embed(dataset)
        if len(x.shape) > 2:
            x = x.reshape((len(dataset), -1))
        return self.model.predict(x)


def _self_optimized_svm_worker(
    storage_url: str,
    study_name: str,
    x: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    n_splits: int,
):
    if n_splits == -1:
        n_splits = len(x)

    def objective(trial: Trial):
        # request hyperparams
        c = trial.suggest_float("C", low=0.01, high=1000, log=True)
        gamma = trial.suggest_float("gamma", low=0.0001, high=10, log=True)
        # do cross-validation
        scores = {"rmse": [], "r2": []}
        for train_index, val_index in KFold(shuffle=True, n_splits=n_splits).split(x):
            train_x = x[train_index]
            train_y = y[train_index]
            val_x = x[val_index]
            val_y = y[val_index]
            model = SVR(C=c, gamma=gamma)
            model.fit(train_x, train_y)
            val_predict = model.predict(val_x)
            scores["rmse"].append(mean_squared_error(val_y, val_predict, squared=False))
            if len(val_y) >= 2:
                scores["r2"].append(r2_score(val_y, val_predict))
        # collect scores
        trial.set_user_attr("scores", scores)
        return np.median(scores["rmse"])

    storage = optuna.storages.RedisStorage(storage_url)
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.load_study(study_name, storage, sampler=sampler)
    study_id = storage.get_study_id_from_name(study_name)
    while storage.get_n_trials(study_id) < n_trials:
        study.optimize(objective, n_trials=1)
