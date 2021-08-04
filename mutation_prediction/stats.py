from typing import Tuple

import numpy as np
from arch.bootstrap import IIDBootstrap, IndependentSamplesBootstrap
from scipy import stats


def confidence_spread(data, alpha):
    n = len(data)
    assert n >= 2
    mean_estimate = np.mean(data)
    variance_estimate = np.sum((data - mean_estimate) ** 2) / (n - 1)
    t = stats.t.ppf(1 - alpha / 2, n - 1)
    return t * np.sqrt(variance_estimate / n)


def needs_repetition(data, alpha, rtol=None, atol=None, min_count=2):
    if len(data) < min_count:
        return True
    c = confidence_spread(data, alpha)
    mean_estimate = np.mean(data)
    if rtol is not None and atol is not None:
        return c > atol + rtol * abs(mean_estimate)
    elif rtol is not None:
        return c > rtol * abs(mean_estimate)
    elif atol is not None:
        return c > atol
    else:
        raise ValueError("At least rtol or atol must be set!")


def mean_difference_spread(
    data_a: np.ndarray, data_b: np.ndarray, alpha: float
) -> Tuple[float, float]:
    n_a = len(data_a)
    n_b = len(data_b)
    diff_estimate = np.mean(data_a) - np.mean(data_b)
    std_estimate = np.sqrt(
        (np.sum((data_a - np.mean(data_a)) ** 2) + np.sum((data_b - np.mean(data_b)) ** 2))
        / (n_a + n_b - 2)
    )
    t = stats.t.ppf(1 - alpha / 2, n_a + n_b - 2)
    confidence = t * std_estimate * np.sqrt(1 / n_a + 1 / n_b)
    return diff_estimate, confidence


def bootstrap(values, metric=np.mean, repetitions=10000, coverage=0.95, method="percentile"):
    values = np.asarray(values)
    estimate = metric(values)
    if len(values) == 0:
        return estimate, estimate, estimate
    bs = IIDBootstrap(np.asarray(values))
    lower, upper = bs.conf_int(metric, reps=repetitions, method=method, size=coverage)
    return lower, estimate, upper


def bootstrap_multiple(metric, *values, repetitions=10000, coverage=0.95, method="percentile"):
    values = [np.asarray(v) for v in values]
    estimate = metric(*values)
    bs = IndependentSamplesBootstrap(*values)
    lower, upper = bs.conf_int(metric, reps=repetitions, method=method, size=coverage)[:, 0]
    return lower, estimate, upper
