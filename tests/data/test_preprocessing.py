import numpy as np
import pytest

import mutation_prediction.data as data
import mutation_prediction.data.preprocessing as preprocessing


def test_shuffle():
    sequence = np.zeros((10,))
    positions = np.zeros((5, 5))
    acids = np.zeros((5, 5))
    num_mutations = np.asarray([3, 2, 0, 1, 5])
    y = np.zeros((5,))
    dataset = data.Dataset(sequence, positions, acids, num_mutations, y)
    assert np.all(
        preprocessing.shuffle(dataset, seed=42).num_mutations == np.asarray([5, 0, 1, 2, 3])
    )


def test_apply_mutations():
    sequence = np.asarray([1, 2, 3])
    positions = np.asarray([[0, 1], [2, 0]])
    acids = np.asarray([[3, 1], [0, 0]])
    num_mutations = np.asarray([2, 1])
    y = np.asarray([0, 0])
    dataset = data.Dataset(sequence, positions, acids, num_mutations, y)
    sequences = dataset.get_sequences()
    assert np.all(sequences == np.asarray([[3, 1, 3], [1, 2, 0]]))


def test_split_by_index():
    sequence = np.zeros((10,))
    positions = np.zeros((5, 5))
    acids = np.zeros((5, 5))
    num_mutations = np.asarray([3, 2, 0, 1, 5])
    y = np.zeros((5,))
    dataset = data.Dataset(sequence, positions, acids, num_mutations, y)
    train, test = preprocessing.split_by_index(dataset, 0.65)
    assert np.all(train.get_num_mutations() == num_mutations[:3])
    assert np.all(test.get_num_mutations() == num_mutations[3:])


def test_split_sample_mutations():
    sequence = np.zeros((10,))
    positions = np.zeros((9, 2))
    acids = np.zeros((9, 2))
    num_mutations = np.asarray([2, 2, 2, 1, 0, 1, 0, 0, 0])
    y = np.zeros((9,))
    dataset = data.Dataset(sequence, positions, acids, num_mutations, y)
    iterations = 10000
    sample_size = 4
    sampled = np.zeros((iterations, sample_size))
    for i in range(iterations):
        train, test = preprocessing.split_sample_mutations(
            dataset, np.asarray([0.9, 0.1, 0]), sample_size
        )
        assert len(train) == sample_size
        sampled[i] = train.get_num_mutations()
    assert np.count_nonzero(sampled == 0) / sampled.size == pytest.approx(0.9, abs=5e-2)
    assert np.count_nonzero(sampled == 1) / sampled.size == pytest.approx(0.1, abs=5e-2)
    assert np.count_nonzero(sampled == 2) == 0


def test_normalizer_1():
    data_a = np.asarray([[2, 0], [4, 2]], dtype=np.float32)
    data_c = np.asarray([[4, 2], [2, 0]], dtype=np.float32)
    maximum = 2
    normalizer = preprocessing.Normalizer()
    normed_a = normalizer.norm_update(data_a)
    assert list(normalizer.maximum) == [pytest.approx(maximum)]
    assert list(normalizer.mean) == [pytest.approx(2)]
    assert np.all(normed_a == (data_a - 2) / maximum)
    normed_c = normalizer.norm(data_c)
    assert np.all(normed_c == (data_c - 2) / maximum)
    denormed_a = normalizer.denorm(normed_a)
    denormed_c = normalizer.denorm(normed_c)
    assert np.all(np.isclose(data_a, denormed_a))
    assert np.all(np.isclose(data_c, denormed_c))


def test_normalizer_2():
    data_a = np.asarray([[2, 0], [4, 2]], dtype=np.float32)
    data_b = np.asarray([[4, 6], [6, 8]], dtype=np.float32)
    data_c = np.asarray([[4, 2], [2, 0]], dtype=np.float32)
    data_d = np.asarray([[6, 8], [4, 6]], dtype=np.float32)
    maximum = 2
    normalizer = preprocessing.Normalizer()
    normed_a, normed_b = normalizer.norm_update(data_a, data_b)
    assert list(normalizer.maximum) == [pytest.approx(maximum), pytest.approx(maximum)]
    assert list(normalizer.mean) == [pytest.approx(2), pytest.approx(6)]
    assert np.all(normed_a == (data_a - 2) / maximum)
    assert np.all(normed_b == (data_b - 6) / maximum)
    normed_c, normed_d = normalizer.norm(data_c, data_d)
    assert np.all(normed_c == (data_c - 2) / maximum)
    assert np.all(normed_d == (data_d - 6) / maximum)
    denormed_a, denormed_b = normalizer.denorm(normed_a, normed_b)
    denormed_c, denormed_d = normalizer.denorm(normed_c, normed_d)
    assert np.all(np.isclose(data_a, denormed_a))
    assert np.all(np.isclose(data_b, denormed_b))
    assert np.all(np.isclose(data_c, denormed_c))
    assert np.all(np.isclose(data_d, denormed_d))


def test_dataset_getitem():
    sequence = np.asarray([1, 2, 3, 4])
    positions = np.asarray([[0, 1], [2, 0], [1, 0]])
    acids = np.asarray([[3, 1], [0, 0], [1, 0]])
    num_mutations = np.asarray([2, 1, 1])
    y = np.asarray([0, -0.5, 1])
    dataset = data.Dataset(sequence, positions, acids, num_mutations, y)
    dataset01 = dataset[0:1]
    assert np.all(dataset.get_sequence() == dataset01.get_sequence())
    assert np.all(dataset01.get_y() == y[0:1])
    assert np.all(dataset01.get_positions() == positions[0:1])
    assert np.all(dataset01.get_acids() == acids[0:1])
    assert np.all(dataset01.get_num_mutations() == num_mutations[0:1])


def test_dataset_add():
    sequence = np.asarray([1, 2, 3])
    positions1 = np.asarray([[0, 1]])
    acids1 = np.asarray([[1, 2]])
    num_mutations1 = np.asarray([2])
    y1 = np.asarray([0.5])
    positions2 = np.asarray([[1]])
    acids2 = np.asarray([[3]])
    num_mutations2 = np.asarray([1])
    y2 = np.asarray([-1.5])
    msa_path = "bar"
    dataset1 = data.Dataset(
        sequence,
        positions1,
        acids1,
        num_mutations1,
        y1,
    )
    dataset2 = data.Dataset(sequence, positions2, acids2, num_mutations2, y2, msa_path=msa_path)
    dataset = dataset1 + dataset2
    assert np.all(dataset.get_sequence() == sequence)
    assert np.all(dataset.get_positions() == np.asarray([[0, 1], [1, 0]]))
    assert np.all(dataset.get_acids() == np.asarray([[1, 2], [3, 0]]))
    assert np.all(dataset.get_num_mutations() == np.asarray([2, 1]))
    assert np.all(dataset.get_y() == np.asarray([0.5, -1.5]))
    assert dataset.get_msa_path() == msa_path


def test_dataset_to_tuples():
    sequence = np.asarray([1, 2, 3, 4])
    positions = np.asarray([[0, 1], [2, 0], [1, 0]])
    acids = np.asarray([[3, 1], [0, 0], [1, 0]])
    num_mutations = np.asarray([2, 1, 1])
    dataset = data.Dataset(sequence, positions, acids, num_mutations)
    tupled = preprocessing.dataset_to_tuples(dataset)
    assert tupled == [{(0, 3), (1, 1)}, {(2, 0)}, {(1, 1)}]


def test_get_mutations():
    sequence = np.asarray([1, 2, 3, 4])
    positions = np.asarray([[0, 1], [2, 0], [1, 0]])
    acids = np.asarray([[3, 1], [0, 0], [1, 0]])
    num_mutations = np.asarray([2, 1, 1])
    dataset = data.Dataset(sequence, positions, acids, num_mutations)
    mutations = preprocessing.get_mutations(dataset)
    assert mutations == {(0, 3), (1, 1), (2, 0)}
