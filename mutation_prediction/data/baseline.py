import csv
from typing import Any, Dict


def load_hyperparameters(variant: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    path = "data/baseline/%s.csv" % variant.lower()
    hyperparameters = {}
    with open(path) as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            params = {}
            for k, v in row.items():
                if k not in ["dataset", "descriptor"]:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                    params[k] = v
            dataset = row["dataset"][len("Public-") :]
            if dataset not in hyperparameters:
                hyperparameters[dataset] = {}
            hyperparameters[dataset][row["descriptor"].strip()] = params
    return hyperparameters


def load_scores() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    scores = {}
    with open("data/baseline/scores.csv") as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            dataset = row["dataset"][len("Public-") :]
            if dataset not in scores:
                scores[dataset] = {}
            descriptor = row["descriptor"].strip()
            if descriptor not in scores[dataset]:
                scores[dataset][descriptor] = {}
            scores[dataset][descriptor][row["method"]] = {
                "RMSE": float(row["RMSE"]),
                "rsq": float(row["rsq"]),
            }
    return scores
