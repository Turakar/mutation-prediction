import optuna
import pytest

from mutation_prediction.models import HyperparameterDict, HyperparameterFloat


def example():
    hyperparams = HyperparameterDict(
        {"a": HyperparameterFloat(), "b": HyperparameterDict({"c": HyperparameterFloat()})}
    )
    params = {"a": {"low": 0, "high": 4}, "b": {"c": {"low": 0, "high": 4}}}

    def objective(trial):
        hyperparams.set_from_trial(trial, params)
        a = hyperparams["a"].get()
        c = hyperparams["b"]["c"].get()
        return a ** 2 - c

    return hyperparams, params, objective


def test_hyperparameter_study():
    _, _, objective = example()

    study = optuna.create_study()
    study.optimize(objective, n_trials=200)
    assert study.best_value == pytest.approx(-4, abs=1e-1)
    assert study.best_params == {"a": pytest.approx(0, abs=1e-1), "b_c": pytest.approx(4, abs=1e-1)}


def test_hyperparameter_set_get():
    hyperparams, _, _ = example()
    with pytest.raises(KeyError):
        hyperparams.set({"a": 1})
    hyperparams.set({"a": 1, "b": {"c": 2}})
    assert hyperparams.get() == {"a": 1, "b": {"c": 2}}
