import functools

import numpy as np
import pytest

import mutation_prediction.data as data
import mutation_prediction.data.loader as loader


def test_a_examples():
    dataset = loader.load_a()
    positions = dataset.get_positions()
    num_mutations = dataset.get_num_mutations()
    acids = dataset.get_acids()
    y = dataset.get_y()
    assert len(dataset.get_sequence()) == 298
    assert positions.max() < len(dataset.get_sequence())
    assert num_mutations[0] == 0
    assert num_mutations[1] == 1
    assert positions[1][0] == 120
    assert acids[1][0] == data.acid_to_index("E")
    assert y[1] == 530
    assert num_mutations[53] == 3
    assert positions[53][0] == 161
    assert acids[53][0] == data.acid_to_index("L")
    assert positions[53][1] == 165
    assert acids[53][1] == data.acid_to_index("W")
    assert positions[53][2] == 255
    assert acids[53][2] == data.acid_to_index("C")
    assert y[53] == 480


def test_b_examples():
    dataset = loader.load_b()
    sequence = dataset.get_sequence()
    positions = dataset.get_positions()
    num_mutations = dataset.get_num_mutations()
    acids = dataset.get_acids()
    y = dataset.get_y()
    assert len(sequence) == 398
    assert positions.max() < len(sequence)
    assert num_mutations[0] == 0
    assert num_mutations[1] == 2
    assert positions[1][0] == 214
    assert acids[1][0] == data.acid_to_index(data.acid_shortname_to_letter("Phe"))
    assert positions[1][1] == 218
    assert acids[1][1] == data.acid_to_index(data.acid_shortname_to_letter("Asp"))
    assert y[1] == 23


def test_c_examples():
    dataset = loader.load_c()
    sequence = dataset.get_sequence()
    positions = dataset.get_positions()
    num_mutations = dataset.get_num_mutations()
    acids = dataset.get_acids()
    y = dataset.get_y()
    assert len(sequence) == 145
    assert positions.max() < len(sequence)
    assert num_mutations[0] == 0
    assert y[0] == 0
    assert num_mutations[1] == 4
    assert positions[1][0] == 31
    assert acids[1][0] == data.acid_to_index("C")
    assert positions[1][1] == 45
    assert acids[1][1] == data.acid_to_index("C")
    assert positions[1][2] == 55
    assert acids[1][2] == data.acid_to_index("I")
    assert positions[1][3] == 96
    assert acids[1][3] == data.acid_to_index("Y")
    assert y[1] == -0.271


def test_d_examples():
    dataset = loader.load_d()
    sequence = dataset.get_sequence()
    positions = dataset.get_positions()
    num_mutations = dataset.get_num_mutations()
    acids = dataset.get_acids()
    y = dataset.get_y()
    assert len(sequence) == 238 - 1  # stop marker excluded
    assert positions.max() < len(sequence)
    assert num_mutations[0] == 0
    assert y[0] == pytest.approx(3.7192121319)
    assert num_mutations[7] == 2
    assert positions[7][0] == 108
    assert acids[7][0] == data.acid_to_index("G")
    assert positions[7][1] == 156
    assert acids[7][1] == data.acid_to_index("R")
    assert y[7] == pytest.approx(3.65901297055)


def test_e_examples():
    train, val, test = loader.load_e()
    assert len(train) == 10
    assert len(val) == 28
    assert np.all(train.num_mutations[1:] == 1)
    assert train.get_positions()[1, 0] == 214
    assert train.get_acids()[1, 0] == data.acid_to_index("F")
    assert train.get_y()[1] == pytest.approx(-1.5)
    assert val.get_num_mutations()[1] == 2
    assert val.get_positions()[1, 0] == 328
    assert val.get_acids()[1, 0] == data.acid_to_index("P")
    assert val.get_positions()[1, 1] == 329
    assert val.get_acids()[1, 1] == data.acid_to_index("Y")
    assert val.get_y()[1] == pytest.approx(-0.87)
    assert test.get_num_mutations()[0] == 3
    assert test.get_positions()[0, 0] == 216
    assert test.get_acids()[0, 0] == data.acid_to_index("N")
    assert test.get_positions()[0, 1] == 218
    assert test.get_acids()[0, 1] == data.acid_to_index("S")
    assert test.get_positions()[0, 2] == 248
    assert test.get_acids()[0, 2] == data.acid_to_index("Y")
    assert test.get_y()[0] == pytest.approx(-1.08)


def test_ids():
    datasets = {
        "A": lambda: loader.load_a(),
        "B": lambda: loader.load_b(),
        "C": lambda: loader.load_c(),
        "D": lambda: loader.load_d(),
        "E": lambda: functools.reduce(lambda a, b: a + b, loader.load_e()),
    }
    for name, load_fn in datasets.items():
        dataset = load_fn()
        assert list(dataset.get_ids()) == sorted(list(dataset.get_ids())), name
