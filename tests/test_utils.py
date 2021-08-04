import numpy as np
import pytest

import mutation_prediction.utils as utils


def test_list_to_matrix():
    lists = [[1], [2, 5], [3]]
    matrix, lengths = utils.list_to_matrix(lists, dtype=np.uint32)
    assert matrix.dtype == np.uint32
    assert np.all(matrix == np.asarray([[1, 0], [2, 5], [3, 0]]))
    assert np.all(lengths == np.asarray([1, 2, 1]))


def test_mask():
    assert np.all(utils.make_mask(5, 1, 3) == np.asarray([False, True, False, True, False]))


def test_merge_dicts_disjoint_example():
    a = {"a": 1, "b": {"c": 2}}
    b = {"d": 3, "b": {"e": 4}}
    merged = utils.merge_dicts_disjoint(a, b)
    assert merged == {"a": 1, "b": {"c": 2, "e": 4}, "d": 3}


def test_merge_dicts_disjoint_exception():
    a = {"a": {"b": 1}}
    b = {"a": {"b": 1}}
    with pytest.raises(RuntimeError, match="Duplicate keys!"):
        utils.merge_dicts_disjoint(a, b)


def test_dynamic_array():
    array = utils.DynamicArray(element_shape=(2,))
    for i in range(1000):
        array.add(np.asarray([i, -i]))
    assert len(array) == 1000
    array.trim()
    assert array[100, 1] == -100
    assert np.allclose(array, np.asarray([np.arange(1000), -np.arange(1000)]).T)


def test_stream_stats():
    shape = (2, 3)
    means = np.asarray([[1, -1, -100], [100, 0, 2.2]])
    stds = np.asarray([[1, 2, 1], [0.1, 1.9, 1.3]])
    stats = utils.StreamStats(shape)
    rng = np.random.default_rng()
    for i in range(100000):
        stats.add(rng.standard_normal(shape) * stds + means)
    assert np.allclose(stats.mean(), means, atol=1e-2)
    assert np.allclose(stats.std(), stds, atol=1e-2)
