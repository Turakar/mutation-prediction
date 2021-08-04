from __future__ import annotations

from typing import Generator

import numpy as np
from Bio import SeqIO
from Bio.PDB import Structure
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import mutation_prediction.utils as utils


class Dataset:
    def __init__(
        self,
        sequence: np.ndarray,
        positions: np.ndarray,
        acids: np.ndarray,
        num_mutations: np.ndarray,
        y: np.ndarray = None,
        structure: Structure = None,
        msa_path: str = None,
        name: str = None,
        ids: np.ndarray = None,
    ):
        self.sequence = sequence
        self.positions = positions
        self.acids = acids
        self.num_mutations = num_mutations
        self.y = y
        self.structure = structure
        self.msa_path = msa_path
        self.name = name
        self.ids = ids if ids is not None else np.arange(len(num_mutations))

    def get_sequences(self) -> np.ndarray:
        num_mutations = self.get_num_mutations()
        sequence = self.get_sequence()
        positions = self.get_positions()
        acids = self.get_acids()
        if positions.shape[1] == 0:
            return np.copy(sequence)[np.newaxis, ...]
        sequences = np.broadcast_to(
            sequence[None, :], (num_mutations.shape[0], sequence.shape[0])
        ).copy()
        for i, n in enumerate(num_mutations):
            sequences[i, positions[i, :n]] = acids[i, :n]
        return sequences

    def in_batches(self, batch_size: int) -> Generator[Dataset]:
        for i in range(0, len(self), batch_size):
            size = min(len(self) - i, batch_size)
            yield Dataset(
                self.sequence,
                self.positions[i : i + size],
                self.acids[i : i + size],
                self.num_mutations[i : i + size],
                self.y[i : i + size],
                self.structure,
                self.msa_path,
                self.name,
                self.ids[i : i + size],
            )

    def __len__(self):
        return len(self.num_mutations)

    def __add__(self, other: Dataset) -> Dataset:
        assert np.all(self.sequence == other.sequence)
        self_positions_width = self.positions.shape[1]
        other_positions_width = other.positions.shape[1]
        max_positions_width = max(self_positions_width, other_positions_width)
        num_mutations = np.concatenate([self.num_mutations, other.num_mutations])
        combined_length = len(num_mutations)
        positions = np.zeros((combined_length, max_positions_width), dtype=np.uint8)
        positions[: len(self), 0:self_positions_width] = self.positions
        positions[len(self) :, 0:other_positions_width] = other.positions
        acids = np.zeros_like(positions)
        acids[: len(self), 0:self_positions_width] = self.acids
        acids[len(self) :, 0:other_positions_width] = other.acids
        if self.y is not None and other.y is not None:
            y = np.concatenate([self.y, other.y])
        else:
            y = None
        return Dataset(
            self.sequence,
            positions,
            acids,
            num_mutations,
            y,
            utils.merge(self.structure, other.structure),
            utils.merge(self.msa_path, other.msa_path),
            utils.merge(self.name, other.name),
            np.concatenate([self.ids, other.ids]),
        )

    def __getitem__(self, item):
        return Dataset(
            self.sequence,
            self.positions[item],
            self.acids[item],
            self.num_mutations[item],
            utils.index_if_not_none(self.y, item),
            self.structure,
            self.msa_path,
            self.name,
            self.ids[item],
        )

    def strip_y(self) -> Dataset:
        return Dataset(
            self.sequence,
            self.positions,
            self.acids,
            self.num_mutations,
            structure=self.structure,
            msa_path=self.msa_path,
            name=self.name,
            ids=self.ids,
        )

    def get_sequence(self) -> np.ndarray:
        return self.sequence

    def get_positions(self) -> np.ndarray:
        return self.positions

    def get_acids(self) -> np.ndarray:
        return self.acids

    def get_num_mutations(self) -> np.ndarray:
        return self.num_mutations

    def get_y(self) -> np.ndarray:
        return self.y

    def get_structure(self) -> Structure:
        return self.structure

    def get_msa_path(self) -> str:
        return self.msa_path

    def get_name(self) -> str:
        return self.name

    def get_ids(self):
        return self.ids


def acid_shortname_to_letter(shortname: str) -> str:
    return {
        "Arg": "R",
        "His": "H",
        "Lys": "K",
        "Asp": "D",
        "Glu": "E",
        "Ser": "S",
        "Thr": "T",
        "Asn": "N",
        "Gln": "Q",
        "Cys": "C",
        "Sec": "U",
        "Gly": "G",
        "Pro": "P",
        "Ala": "A",
        "Val": "V",
        "Ile": "I",
        "Leu": "L",
        "Met": "M",
        "Phe": "F",
        "Tyr": "Y",
        "Trp": "W",
    }[shortname]


def genome_to_acid(genome: str) -> str:
    return {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "TAT": "Y",
        "TAC": "Y",
        "TGT": "C",
        "TGC": "C",
        "TGG": "W",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGT": "S",
        "AGC": "S",
        "AGG": "R",
        "AGA": "R",
        "GTT": "V",
        "GTC": "V",
        "GTG": "V",
        "GTA": "V",
        "GCT": "A",
        "GCC": "A",
        "GCG": "A",
        "GCA": "A",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }[genome]


def acid_to_index(acid: str) -> int:
    return {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
    }[acid]


def index_to_acid(index: int) -> str:
    if index < 0:
        return "-"
    elif index == num_acids():
        return "X"
    return [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ][index]


def num_acids():
    return 20


def num_genes():
    return 4


def sequence_genome_to_acid(genome_sequence: str) -> str:
    acid_sequence = ""
    assert len(genome_sequence) % 3 == 0
    for i in range(0, len(genome_sequence), 3):
        acid_sequence += genome_to_acid(genome_sequence[i : i + 3])
    return acid_sequence


def sequence_to_string(sequence: np.ndarray) -> str:
    return "".join([index_to_acid(i) for i in sequence])


def write_to_fasta(fd, sequences: np.ndarray, id: str = "%d"):
    for i, sequence in enumerate(sequences):
        seq = Seq(sequence_to_string(sequence))
        if "%d" in id:
            name = id % i
        else:
            name = id
        record = SeqRecord(seq, id=name)
        SeqIO.write(record, fd, "fasta")


def mutant_to_str(dataset: Dataset, index: int) -> str:
    singles = []
    for i in range(dataset.get_num_mutations()[index]):
        singles.append(
            "%s%d%s"
            % (
                index_to_acid(dataset.get_sequence()[dataset.get_positions()[index, i]]),
                dataset.get_positions()[index, i] + 1,
                index_to_acid(dataset.get_acids()[index, i]),
            )
        )
    return "+".join(singles)
