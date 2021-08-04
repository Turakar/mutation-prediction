import functools
import re
from typing import List, Set, Tuple, Union

import numpy as np

import mutation_prediction.data as data


def shuffle(dataset: data.Dataset, seed=None):
    if seed is not None:
        rng = np.random.Generator(np.random.PCG64(seed))
    else:
        rng = np.random.default_rng()
    permutation = rng.permutation((len(dataset)))
    shuffled = dataset[permutation]
    return shuffled


def split_by_mask(dataset: data.Dataset, mask: np.ndarray) -> Tuple[data.Dataset, data.Dataset]:
    train = dataset[mask]
    test = dataset[~mask]
    return train, test


def split_by_index(
    dataset: data.Dataset, training_fraction: float
) -> Tuple[data.Dataset, data.Dataset]:
    mask = np.zeros((len(dataset),), dtype=np.bool8)
    pivot = int(training_fraction * len(dataset))
    mask[0:pivot] = True
    return split_by_mask(dataset, mask)


def split_by_index_num(dataset: data.Dataset, num: int) -> Tuple[data.Dataset, data.Dataset]:
    mask = np.zeros((len(dataset),), dtype=np.bool8)
    mask[0:num] = True
    return split_by_mask(dataset, mask)


def split_sample_mutations(
    dataset: data.Dataset, probabilities: np.ndarray, num: int
) -> Tuple[data.Dataset, data.Dataset]:
    num_mutations = dataset.get_num_mutations()
    p = probabilities[num_mutations]
    p = p / p.sum()
    chosen = np.random.choice(len(num_mutations), size=num, replace=False, p=p)
    mask = np.zeros_like(num_mutations, dtype=np.bool8)
    mask[chosen] = True
    return split_by_mask(dataset, mask)


def split_num_mutations(
    dataset: data.Dataset, max_mutations: int
) -> Tuple[data.Dataset, data.Dataset]:
    mask = dataset.get_num_mutations() <= max_mutations
    return split_by_mask(dataset, mask)


def split_single_to_multiple(
    dataset: data.Dataset, mutations: List[str]
) -> Tuple[data.Dataset, data.Dataset]:
    mutations_set = set()
    for mutation in mutations:
        match = re.match("([A-Z])([0-9]+)([A-Z])", mutation)
        old_acid = data.acid_to_index(match.group(1))
        position = int(match.group(2)) - 1
        new_acid = data.acid_to_index(match.group(3))
        assert dataset.get_sequence()[position] == old_acid
        mutations_set.add((position, new_acid))
    train_indices = []
    test_indices = []
    for i in range(len(dataset)):
        all_in = True
        for j in range(dataset.get_num_mutations()[i]):
            mutation = (dataset.get_positions()[i, j], dataset.get_acids()[i, j])
            if mutation not in mutations_set:
                all_in = False
        if all_in:
            if dataset.get_num_mutations()[i] <= 1:
                train_indices.append(i)
            else:
                test_indices.append(i)
    return dataset[train_indices], dataset[test_indices]


class Normalizer:
    def __init__(self):
        self.mean = None
        self.maximum = None

    def norm_update(self, *xs: np.ndarray) -> Tuple[np.ndarray]:
        mean = np.zeros((len(xs),), dtype=np.float32)
        maximum = np.zeros((len(xs),), dtype=np.float32)
        for i, x in enumerate(xs):
            mean[i] = x.mean()
            maximum[i] = np.abs(x - mean[i]).max()
        self.mean = mean
        self.maximum = maximum
        return self.norm(*xs)

    def norm(self, *xs: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray]]:
        normed = []
        for i, x in enumerate(xs):
            if self.maximum[i] != 0:
                normed.append((x - self.mean[i]) / self.maximum[i])
            else:
                normed.append(x - self.mean[i])
        if len(normed) == 1:
            return normed[0]
        else:
            return tuple(normed)

    def denorm(self, *xs: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray]]:
        denormed = []
        for i, x in enumerate(xs):
            denormed.append(x * self.maximum[i] + self.mean[i])
        if len(denormed) == 1:
            return denormed[0]
        else:
            return tuple(denormed)


def dataset_to_tuples(dataset: data.Dataset) -> List[Set[Tuple[int, int]]]:
    tupled = []
    for i in range(len(dataset)):
        mutant = set()
        for j in range(dataset.get_num_mutations()[i]):
            mutant.add((dataset.get_positions()[i, j], dataset.get_acids()[i, j]))
        tupled.append(mutant)
    return tupled


def get_mutations(dataset: data.Dataset) -> Set[Tuple[int, int]]:
    tupled = dataset_to_tuples(dataset)
    return functools.reduce(lambda a, b: a.union(b), tupled)
