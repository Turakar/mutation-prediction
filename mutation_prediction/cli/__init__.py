import argparse
import logging
import sys
import time
import warnings

import hjson
import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

import mutation_prediction.cli.registry as registry
import mutation_prediction.data.baseline as baseline
import mutation_prediction.data.preprocessing as preprocessing
from mutation_prediction import utils
from mutation_prediction.models import (
    ModelObjectiveCrossValidation,
    ModelObjectiveFixedValidation,
    ModelObjectiveSelfScoring,
)


def main():
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore", message="The dataloader, train dataloader, does not have many workers.*"
    )
    warnings.filterwarnings(
        "ignore", message="The dataloader, val dataloader 0, does not have many workers.*"
    )

    parser = argparse.ArgumentParser(
        description="Mutation Prediction optimization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", "-d", help="SQLAlchemy DB URL")
    subparsers = parser.add_subparsers(title="subcommands", required=True, dest="subcommand")

    parser_new = subparsers.add_parser(
        "new", help="Create a new study or overwrite an existing one."
    )
    parser_new.set_defaults(func=main_new)
    parser_new.add_argument("name")
    parser_new.add_argument("model")
    parser_new.add_argument("dataset")
    parser_new.add_argument("params", nargs="+")
    parser_new.add_argument(
        "--sampler",
        "-s",
        help="Optuna sampler to use.",
        choices=["RandomSampler", "TPESampler", "CmaEsSampler"],
        default="TPESampler",
    )
    parser_new.add_argument(
        "--multivariate",
        action="store_true",
        help="Make TPE Sampler use a multivariate Parzen estimator instead of multiple independent ones.",
    )
    parser_new.add_argument(
        "--startup-trials",
        type=int,
        help="Number of random samples before using TPESampler/CmaEsSampler.",
    )
    parser_new.add_argument(
        "--group",
        action="store_true",
        help="Make TPE sampler use grouped multivariate distributions for "
        "conditional parameters.",
    )
    parser_new.add_argument(
        "--liar",
        action="store_true",
        help="Make TPE sampler use the constant liar feature to avoid evaluating similar"
        " hyperparameter configurations at once.",
    )
    parser_new.add_argument(
        "--objective",
        "-o",
        default="cross-validation",
        choices=["cross-validation", "fixed", "self-scoring"],
        help="Which model objective function to use.",
    )
    parser_new.add_argument(
        "--splits",
        "-k",
        default=5,
        type=int,
        help="Number of splits used in case of cross-validation.",
    )
    parser_new.add_argument(
        "--iterations",
        "-i",
        default=5,
        type=int,
        help="Number of iterations used in case of fixed validation set.",
    )

    parser_optimize = subparsers.add_parser("optimize", help="Optimize an existing study.")
    parser_optimize.set_defaults(func=main_optimize)
    parser_optimize.add_argument("name")
    parser_optimize.add_argument(
        "--trials",
        "-n",
        help="If the specified number of trials is reached in total, optimization stops.",
        type=int,
    )
    parser_optimize.add_argument(
        "--timeout",
        "-t",
        help="If the specified number of seconds passed, optimization stops.",
        type=float,
    )
    parser_optimize.add_argument(
        "--early-stop",
        "-e",
        type=int,
        help="If the specified number of trials give no improvement, optimization stops.",
    )
    parser_optimize.add_argument(
        "--iterations",
        "-i",
        help="If the specified number of trials is reached for this process, optimization stops.",
        type=int,
    )

    parser_reevaluate = subparsers.add_parser(
        "reevaluate", help="Re-evaluate a cross-validation objective for better estimation."
    )
    parser_reevaluate.set_defaults(func=main_reevaluate)
    parser_reevaluate.add_argument("name")
    parser_reevaluate.add_argument("--top", type=int)
    parser_reevaluate.add_argument("--trials", type=str)

    parser_evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate the best hyperparameter configuration on the test set."
    )
    parser_evaluate.set_defaults(func=main_evaluate)
    parser_evaluate.add_argument("name")
    parser_evaluate.add_argument(
        "--iterations", "-i", help="Number of iterations to take median over.", default=1, type=int
    )
    parser_evaluate.add_argument(
        "--trial", "-t", help="Trial number to evaluate. Use best trial if not set.", type=int
    )
    parser_evaluate.add_argument("--dataset", help="Specify a different dataset for evaluation.")

    parser_copy = subparsers.add_parser(
        "copy", help="Copy a study between two storages with optional renaming."
    )
    parser_copy.set_defaults(func=main_copy)
    parser_copy.add_argument("name")
    parser_copy.add_argument("source")
    parser_copy.add_argument("destination")
    parser_copy.add_argument("--rename", type=str, help="The new name of the study.", default=None)
    parser_copy.add_argument("--yes", "-y", action="store_true")

    args = parser.parse_args()
    args.func(args)


def main_new(args):
    params_hjson = " ".join(args.params)
    try:
        params = hjson.loads(params_hjson)
    except hjson.scanner.HjsonDecodeError as e:
        print("Invalid HJSON!")
        print(e)
        print(params_hjson)
        sys.exit(1)

    if args.model not in registry.models:
        raise KeyError("Unknown model! %s" % args.model)
    if args.dataset not in registry.datasets:
        raise KeyError("Unknown dataset! %s" % args.dataset)
    sampler_args = {}
    if args.multivariate:
        sampler_args["multivariate"] = True
    if args.startup_trials:
        sampler_args["n_startup_trials"] = args.startup_trials
    if args.group:
        sampler_args["group"] = True
    if args.liar:
        sampler_args["constant_liar"] = True
    sampler = get_sampler(args.sampler, sampler_args)

    storage = get_storage(args)
    study = optuna.create_study(
        storage=storage, study_name=args.name, load_if_exists=False, sampler=sampler
    )
    study.set_user_attr("params", params)
    study.set_user_attr("model", args.model)
    study.set_user_attr("dataset", args.dataset)
    study.set_user_attr("objective", args.objective)
    if args.objective == "cross-validation":
        study.set_user_attr("splits", args.splits)
    elif args.objective == "fixed":
        study.set_user_attr("iterations", args.iterations)
    study.set_user_attr("sampler", args.sampler)
    study.set_user_attr("sampler_args", sampler_args)


def main_optimize(args):
    if (
        args.early_stop is None
        and args.trials is None
        and args.timeout is None
        and args.iterations is None
    ):
        raise RuntimeError("No stopping criterion given!")

    model, params, study, datasets, _ = load_study(args)

    # create objective
    objective = make_objective(datasets, model, params, study)

    # start optimization
    start_time = time.time()
    iteration = 0
    while True:

        if (
            args.trials is not None
            and len(
                [
                    trial
                    for trial in study.trials
                    if trial.state
                    in [
                        TrialState.COMPLETE,
                        TrialState.RUNNING,
                        TrialState.PRUNED,
                    ]
                ]
            )
            >= args.trials
        ):
            print("Maximum number of total trials reached.")
            break
        if args.timeout is not None and time.time() > start_time + args.timeout:
            print("Timeout reached.")
            break

        study.optimize(objective, n_trials=1)
        last_trial = study.trials[-1]
        iteration += 1

        if (
            args.early_stop is not None
            and last_trial.number >= study.best_trial.number + args.early_stop
        ):
            print("Early stopping criterion reached.")
            break

        if args.iterations is not None and iteration >= args.iterations:
            print("Maximum number of trials for this process reached.")
            break


def main_reevaluate(args):
    # initialize
    model, params, study, datasets, storage = load_study(args)
    assert study.user_attrs["objective"] == "cross-validation"
    dataset, _ = datasets

    # select trials to re-evaluate
    if args.top:
        all_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
        trials = sorted(all_trials, key=lambda trial: trial.value)[: args.top]
    elif args.trials:
        trial_numbers = {int(x) for x in args.trials.split(",")}
        trials = [trial for trial in study.trials if trial.number in trial_numbers]
    else:
        raise ValueError("Neither --top nor --trials specified!")
    reevaluation = study.user_attrs.get("reevaluation", {})
    trials = [trial for trial in trials if str(trial.number) not in reevaluation]

    # re-evaluate trials
    for trial in tqdm(trials):
        # collect leave-one-out-cross-validation predictions
        predictions = np.zeros((len(dataset),), dtype=dataset.get_y().dtype)
        for val_idx in tqdm(range(len(dataset)), leave=False):
            mask = utils.make_mask(len(dataset), val_idx)
            train = dataset[~mask]
            val = dataset[mask]
            model.hyperparams.set_from_trial(trial, params)
            model.fit(train)
            val_pred = model.predict(val)
            predictions[val_idx] = val_pred[0]

        # update scores
        reevaluation[trial.number] = {
            "rmse": mean_squared_error(dataset.get_y(), predictions, squared=False),
            "r2": r2_score(dataset.get_y(), predictions),
        }
        study.set_user_attr("reevaluation", reevaluation)


def main_evaluate(args):
    model, params, study, datasets, storage = load_study(args)
    if args.trial is not None:
        trial = study.trials[args.trial]
    elif "selected" in study.user_attrs:
        trial = study.trials[study.user_attrs["selected"]]
    else:
        trial = study.best_trial
    print("Evaluating trial #%d" % trial.number)

    iterations = args.iterations

    evaluation_key = "evaluation"
    if args.dataset and args.dataset != study.user_attrs["dataset"]:
        evaluation_key = "evaluation_%s" % args.dataset
    evaluation = study.user_attrs.get(evaluation_key, {})
    if str(trial.number) in evaluation:
        scores = evaluation[str(trial.number)]
        iterations -= len(scores["test_rmse"])
    else:
        scores = {
            "train_rmse": [],
            "train_r2": [],
            "test_rmse": [],
            "test_r2": [],
        }
    model.hyperparams.set_from_trial(trial, params)

    if iterations > 0:
        objective = trial.user_attrs.get("objective") or "cross-validation"
        if objective in ["cross-validation", "self-scoring"]:
            if len(datasets) == 3:
                train = preprocessing.shuffle(datasets[0] + datasets[1])
            else:
                train = datasets[0]
        elif objective == "fixed":
            train = datasets[0]
        else:
            raise KeyError("Unknown objective!")
        test = datasets[-1]

        iteration_generator = range(iterations)
        if args.iterations > 1:
            iteration_generator = tqdm(iteration_generator)
        for _ in iteration_generator:
            model.fit(train, trial=trial)
            if len(test) >= 1000:
                prediction = model.predict_batched(test, 1000)
            else:
                prediction = model.predict(test)
            scores["test_rmse"].append(
                metrics.mean_squared_error(test.get_y(), prediction, squared=False)
            )
            scores["test_r2"].append(metrics.r2_score(test.get_y(), prediction))
            if len(train) >= 1000:
                prediction = model.predict_batched(train, 1000)
            else:
                prediction = model.predict(train)
            scores["train_rmse"].append(
                metrics.mean_squared_error(train.get_y(), prediction, squared=False)
            )
            scores["train_r2"].append(metrics.r2_score(train.get_y(), prediction))

            evaluation = study.user_attrs.get(evaluation_key, {})
            evaluation[str(trial.number)] = scores
            study.set_user_attr(evaluation_key, evaluation)

    rmse = float(np.median(scores["test_rmse"]))
    r2 = float(np.median(scores["test_r2"]))

    print("RMSE test values: %s" % str(scores["test_rmse"]))
    print("R² test values: %s" % str(scores["test_r2"]))
    print("RMSE train values: %s" % str(scores["train_rmse"]))
    print("R² train values: %s" % str(scores["train_r2"]))
    print("Trial value: %.3f" % trial.value)

    print(hjson.dumps(model.hyperparams.get()))
    dataset: str = study.user_attrs["dataset"]
    model_name: str = study.user_attrs["model"]
    if model_name.startswith("Base") and dataset in ["A", "B", "C", "D"]:
        method, descriptor = tuple(model_name[len("Base") :].split("_"))
        base_params = baseline.load_hyperparameters(method)[dataset][descriptor]
        print(hjson.dumps(base_params))
        base_scores = baseline.load_scores()[dataset][descriptor][method]
        print(
            "%s: %.3f (%.3f) / %.3f (%.3f)"
            % (args.name, rmse, base_scores["RMSE"], r2, base_scores["r2"])
        )
    else:
        print("%s: %.3f / %.3f" % (args.name, rmse, r2))


def main_copy(args):
    from_storage = url_to_storage(args.source)
    from_name = args.name
    to_storage = url_to_storage(args.destination)
    to_name = args.rename
    if to_name is None:
        to_name = from_name
    try:
        optuna.load_study(to_name, to_storage)
        if args.yes or input("Study already exists in destination. Overwrite? [y/n] ") == "y":
            optuna.delete_study(to_name, to_storage)
        else:
            return
    except KeyError:
        pass
    optuna.copy_study(
        from_name,
        from_storage,
        to_storage,
        to_name,
    )


def load_study(args):
    storage = get_storage(args)
    study_id = storage.get_study_id_from_name(args.name)
    study_user_attrs = storage.get_study_user_attrs(study_id)
    sampler_name = study_user_attrs.get("sampler") or "TPESampler"
    sampler_args = study_user_attrs.get("sampler_args") or {}
    sampler = get_sampler(sampler_name, sampler_args)
    study = optuna.create_study(
        storage=storage, sampler=sampler, study_name=args.name, load_if_exists=True
    )
    model = registry.models[study.user_attrs["model"]]()
    params = study.user_attrs["params"]
    datasets = registry.datasets[
        args.dataset
        if hasattr(args, "dataset") and args.dataset is not None
        else study.user_attrs["dataset"]
    ]()
    datasets = tuple(preprocessing.shuffle(d) for d in datasets)
    return model, params, study, datasets, storage


def make_objective(datasets, model, params, study, splits=None):
    objective = None
    if "objective" in study.user_attrs:
        objective_name = study.user_attrs["objective"]
    else:
        objective_name = "cross-validation"
    if objective_name in ["cross-validation", "self-scoring"]:
        if len(datasets) == 3:
            train = preprocessing.shuffle(datasets[0] + datasets[1])
        else:
            train = datasets[0]
        if objective_name == "cross-validation":
            objective = ModelObjectiveCrossValidation(
                model,
                params,
                train,
                splits=splits if splits is not None else study.user_attrs["splits"],
            )
        elif objective_name == "self-scoring":
            objective = ModelObjectiveSelfScoring(model, params, train)
    elif objective_name == "fixed":
        objective = ModelObjectiveFixedValidation(
            model, params, datasets[0], datasets[1], iterations=study.user_attrs["iterations"]
        )
    if objective is None:
        raise KeyError("Unknown objective!")
    return objective


def get_sampler(sampler_name, sampler_args):
    sampler_constructor = {
        "TPESampler": optuna.samplers.TPESampler,
        "RandomSampler": optuna.samplers.RandomSampler,
        "CmaEsSampler": optuna.samplers.CmaEsSampler,
    }[sampler_name]
    sampler = sampler_constructor(**sampler_args)
    return sampler


def get_storage(args) -> optuna.storages.BaseStorage:
    return url_to_storage(args.db)


def url_to_storage(url: str) -> optuna.storages.BaseStorage:
    if url.startswith("redis://") or url.startswith("unix://"):
        return optuna.storages.RedisStorage(url)
    else:
        return optuna.storages.RDBStorage(
            url=url, heartbeat_interval=60, engine_kwargs=dict(pool_pre_ping=True)
        )
