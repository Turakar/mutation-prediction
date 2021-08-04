import copy
from typing import Any, Dict, List, Tuple

import numpy as np


def list_to_matrix(lists: List[List[Any]], dtype=np.uint8) -> Tuple[np.ndarray, np.ndarray]:
    n = len(lists)
    m = max(map(len, lists))
    lengths = np.zeros((n,), dtype=np.uint64)
    matrix = np.zeros((n, m), dtype=dtype)
    for i in range(n):
        ls = lists[i]
        lengths[i] = len(ls)
        for j in range(lengths[i]):
            matrix[i, j] = ls[j]
    return matrix, lengths


def make_mask(size, *indices) -> np.ndarray:
    mask = np.zeros((size,), dtype=np.bool8)
    if len(indices) > 0:
        mask[np.asarray(indices)] = True
    return mask


def merge_dicts_disjoint(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.copy(a)
    for key, value in b.items():
        if key not in merged:
            merged[key] = value
        elif type(value) is dict:
            merged[key] = merge_dicts_disjoint(a[key], value)
        else:
            raise RuntimeError("Duplicate keys!")
    return merged


def conv1d_output_length(
    input_length: int, kernel_size: int, stride: int, padding: int, dilation: int
) -> int:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    return int((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


class DynamicArray:
    def __init__(
        self, initial_size=8, element_shape=(), dtype=np.float32, growth_factor=1.125, refcheck=True
    ):
        self.data = np.zeros((initial_size,) + element_shape, dtype=dtype)
        self.pointer = 0
        self.growth_factor = growth_factor
        self.element_shape = element_shape
        self.refcheck = refcheck

    def __len__(self) -> int:
        return self.pointer

    def __getitem__(self, item: Any):
        return self.get()[item]

    def get(self) -> np.ndarray:
        return self.data[0 : self.pointer]

    def add(self, element: Any):
        if self.pointer == len(self.data):
            self.data.resize(
                (int(len(self.data) * self.growth_factor),) + self.element_shape,
                refcheck=self.refcheck,
            )
        self.data[self.pointer] = element
        self.pointer += 1

    def trim(self):
        self.data.resize((self.pointer,) + self.element_shape, refcheck=self.refcheck)


def merge(a: Any, b: Any) -> Any:
    if a is not None:
        return a
    else:
        return b


def index_if_not_none(a: Any, index: Any) -> Any:
    if a is not None:
        return a[index]
    else:
        return a


class StreamStats:
    """
    Computes statistic values based on B. P. Welford's algorithm published 1962.
    """

    def __init__(self, shape: Tuple[int, ...], dtype=np.float64):
        self.n: int = 0
        self.m = np.zeros(shape, dtype=dtype)
        self.s = np.zeros(shape, dtype=dtype)

    def add(self, value: np.ndarray):
        # See Knuth TAOCP vol 2, 3rd edition, page 232
        assert value.shape == self.m.shape
        self.n += 1
        if self.n == 1:
            self.m[:] = value
            self.s = 0
        else:
            diff = value - self.m
            self.m = self.m + diff / self.n
            self.s = self.s + diff * (value - self.m)

    def count(self) -> int:
        return self.n

    def mean(self):
        assert self.n >= 1
        return self.m

    def variance(self):
        assert self.n >= 2
        return self.s / (self.n - 1)

    def std(self):
        return np.sqrt(self.variance())


class SetHyperparameter:
    def __init__(self, model, param, value):
        self.model = model
        self.param = param
        self.value = value
        self.old_value = None

    def __enter__(self):
        self.old_value = self.model.hyperparams[self.param].get()
        self.model.hyperparams[self.param].set(self.value)

    def __exit__(self, *args):
        self.model.hyperparams[self.param].set(self.old_value)


def rank(array: np.ndarray) -> np.ndarray:
    sorting = np.argsort(array)
    ranks = np.empty_like(sorting)
    ranks[sorting] = np.arange(len(array)) + 1
    return ranks
