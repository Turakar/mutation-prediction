import csv
import re
from os.path import join as path_join
from typing import Any, Dict, List, Tuple

import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser

import mutation_prediction.data as data
import mutation_prediction.utils as utils


def load_a(path="data/datasets/A", y_key="lmax") -> data.Dataset:
    sequence = _load_a_sequence(path)
    positions = []
    acids = []
    y = []
    with open(path_join(path, "mutations.csv")) as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            p, a = _parse_a_mutations(sequence, row["variant"])
            positions.append(p)
            acids.append(a)
            y.append(float(row[y_key]))
    positions, num_mutations = utils.list_to_matrix(positions, dtype=np.uint64)
    acids, _ = utils.list_to_matrix(acids)
    y = np.asarray(y)
    structure = MMCIFParser(QUIET=True).get_structure("6nwd", path_join(path, "6nwd_updated.cif"))
    return data.Dataset(
        sequence,
        positions,
        acids,
        num_mutations,
        y,
        structure,
        path_join(path, "hhblits.tar.gz"),
        "A",
    )


def _load_a_sequence(path: str) -> np.ndarray:
    with open(path_join(path, "base-genom.txt")) as fd:
        genome = fd.read().upper()
        sequence = np.zeros((int(len(genome) / 3),), dtype=np.uint8)
        for i in range(sequence.shape[0]):
            part = genome[i * 3 : (i + 1) * 3]
            sequence[i] = data.acid_to_index(data.genome_to_acid(part))
        return sequence


def _parse_a_mutations(sequence: np.ndarray, mutations: str) -> Tuple[List[int], List[int]]:
    if mutations == "WT":
        return [], []
    positions = []
    acids = []
    for part in mutations.split("/"):
        match = re.match("([A-Z])([0-9]+)([A-Z])", part)
        original = match.group(1)
        position = int(match.group(2)) - 1
        assert sequence[position] == data.acid_to_index(original)
        positions.append(position)
        acids.append(data.acid_to_index(match.group(3)))
    return positions, acids


def load_b(
    path="data/datasets/B",
) -> data.Dataset:
    sequence = _load_b_sequence(path)
    positions = []
    acids = []
    y = []
    with open(path_join(path, "data2.csv")) as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            if row["id"] == "WT":
                p = []
                a = []
            else:
                p, a = _parse_b_mutations(sequence, row)
            positions.append(p)
            acids.append(a)
            y.append(float(row["E"]))
    positions, num_mutations = utils.list_to_matrix(positions, dtype=np.uint64)
    acids, _ = utils.list_to_matrix(acids)
    y = np.asarray(y)
    structure = MMCIFParser(QUIET=True).get_structure("3g0i", path_join(path, "3g0i_updated.cif"))
    return data.Dataset(
        sequence,
        positions,
        acids,
        num_mutations,
        y,
        structure,
        path_join(path, "hhblits.tar.gz"),
        "B",
    )


def _load_b_sequence(path: str) -> np.ndarray:
    with open(path_join(path, "sequence2.txt")) as fd:
        sequence = fd.read()
        sequence = list(map(data.acid_to_index, sequence))
        return np.asarray(sequence)


def _parse_b_mutations(sequence: np.ndarray, row: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    positions = []
    acids = []
    # for match in re.finditer("([A-Z][a-z]{2}) ([0-9]+)", mutations):
    for mutation in [v for k, v in row.items() if k.startswith("mutation_")]:
        if "-" not in mutation and len(mutation) != 0:
            match = re.match("([A-Z][a-z]{2}) ([0-9]+)", mutation)
            p = int(match.group(2)) - 1
            a = data.acid_to_index(data.acid_shortname_to_letter(match.group(1)))
            if sequence[p] != a:
                positions.append(p)
                acids.append(a)
    return positions, acids


def load_c(
    path="data/datasets/C",
) -> data.Dataset:
    sequence = _load_c_sequence(path)
    positions = [[]]
    acids = [[]]
    y = [0]
    with open(path_join(path, "data.csv")) as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            p, a = _parse_c_mutations(sequence, row["Description"])
            positions.append(p)
            acids.append(a)
            y.append(float(row["Data"]))
    positions, num_mutations = utils.list_to_matrix(positions, dtype=np.uint64)
    acids, _ = utils.list_to_matrix(acids)
    y = np.asarray(y)
    structure = MMCIFParser(QUIET=True).get_structure("6wk3", path_join(path, "6wk3_updated.cif"))
    return data.Dataset(
        sequence,
        positions,
        acids,
        num_mutations,
        y,
        structure,
        path_join(path, "hhblits.tar.gz"),
        "C",
    )


def _load_c_sequence(path: str) -> np.ndarray:
    with open(path_join(path, "sequence.txt")) as fd:
        sequence = fd.read()
        sequence = list(map(data.acid_to_index, sequence))
        return np.asarray(sequence)


def _parse_c_mutations(sequence: np.ndarray, mutations: str) -> Tuple[List[int], List[int]]:
    positions = []
    acids = []
    for part in mutations.split("+"):
        match = re.match("([A-Z])([0-9]+)([A-Z])", part)
        o = match.group(1)
        p = int(match.group(2)) - 1
        assert sequence[p] == data.acid_to_index(o)
        a = data.acid_to_index(match.group(3))
        if sequence[p] != a:
            positions.append(p)
            acids.append(a)
    return positions, acids


def load_d(
    path="data/datasets/D",
) -> data.Dataset:
    sequence = _load_d_sequence(path)
    positions = []
    acids = []
    y = []
    with open(path_join(path, "amino_acid_genotypes_to_brightness.tsv")) as fd:
        reader = csv.DictReader(fd, delimiter="\t")
        for row in reader:
            if "*" not in row["aaMutations"]:
                p, a = _parse_d_mutations(sequence, row["aaMutations"])
                positions.append(p)
                acids.append(a)
                y.append(float(row["medianBrightness"]))
    positions, num_mutations = utils.list_to_matrix(positions, dtype=np.uint64)
    acids, _ = utils.list_to_matrix(acids)
    y = np.asarray(y)
    structure = MMCIFParser(QUIET=True).get_structure(
        "5n9o", path_join(path, "structure_candidates", "5n9o.cif")
    )
    return data.Dataset(
        sequence,
        positions,
        acids,
        num_mutations,
        y,
        structure,
        path_join(path, "hhblits.tar.gz"),
        "D",
    )


def _load_d_sequence(path: str) -> np.ndarray:
    with open(path_join(path, "sequence.txt")) as fd:
        genome = fd.read()
        sequence = np.zeros((int(len(genome) / 3),), dtype=np.uint8)
        for i in range(sequence.shape[0]):
            part = genome[i * 3 : (i + 1) * 3]
            sequence[i] = data.acid_to_index(data.genome_to_acid(part))
        return sequence


def _parse_d_mutations(sequence: np.ndarray, mutations: str) -> Tuple[List[int], List[int]]:
    if mutations == "":
        return [], []
    positions = []
    acids = []
    for part in mutations.split(":"):
        match = re.match("S([A-Z])([0-9]+)([A-Z])", part)
        if not match:
            raise ValueError("%s / %s" % (mutations.split(":"), part))
        o = match.group(1)
        p = int(match.group(2))
        assert sequence[p] == data.acid_to_index(o), part
        a = data.acid_to_index(match.group(3))
        positions.append(p)
        acids.append(a)
    return positions, acids


def load_e(path="data/datasets/E", y_key="ddG") -> Tuple[data.Dataset, data.Dataset, data.Dataset]:
    sequence = _load_e_sequence(path)
    structure = MMCIFParser(QUIET=True).get_structure("3g0i", path_join(path, "3g0i_updated.cif"))

    def load_file(name, from_id):

        positions = []
        acids = []
        y = []
        with open(path_join(path, name)) as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                p, a = _parse_e_mutations(sequence, row["mutations"])
                positions.append(p)
                acids.append(a)
                y.append(float(row[y_key]))
        positions, num_mutations = utils.list_to_matrix(positions, dtype=np.uint64)
        acids, _ = utils.list_to_matrix(acids)
        y = np.asarray(y)
        return data.Dataset(
            sequence,
            positions,
            acids,
            num_mutations,
            y,
            structure=structure,
            msa_path=path_join(path, "hhblits.tar.gz"),
            name="E",
            ids=np.arange(len(num_mutations)) + from_id,
        )

    train = load_file("train.csv", 0)
    val = load_file("validation.csv", len(train))
    test = load_file("test.csv", len(train) + len(val))
    return train, val, test


def _parse_e_mutations(sequence: np.ndarray, mutations: str) -> Tuple[List[int], List[int]]:
    if mutations == "WT":
        return [], []
    positions = []
    acids = []
    for part in mutations.split("_"):
        match = re.match("([A-Z])([0-9]+)([A-Z])", part)
        original = match.group(1)
        position = int(match.group(2)) - 1
        assert sequence[position] == data.acid_to_index(original)
        positions.append(position)
        acids.append(data.acid_to_index(match.group(3)))
    return positions, acids


def _load_e_sequence(path: str) -> np.ndarray:
    with open(path_join(path, "sequence.txt")) as fd:
        sequence = fd.read()
        sequence = list(map(data.acid_to_index, sequence))
        return np.asarray(sequence)
