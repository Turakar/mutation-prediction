from typing import Dict, Set, Tuple

import numpy as np

import mutation_prediction_native as native
from mutation_prediction import data
from mutation_prediction.data import Dataset


def load_alignments(dataset: Dataset) -> Dict[str, Tuple[str, Set[str]]]:
    return native.load_alignments(dataset.get_msa_path(), dataset.get_ids(), dataset.get_name())


def make_msa(dataset: Dataset) -> np.ndarray:
    alignments = load_alignments(dataset)
    ids = sorted(list(alignments.keys()))
    sequence = dataset.get_sequence()
    msa = np.zeros((len(ids), len(sequence)), dtype=np.int8)
    for i, id in enumerate(ids):
        alignment = alignments[id][0]
        j = 0  # counts index in original sequence
        for acid in alignment:
            # we ignore insert, i.e. lower case letters, here
            if acid.isupper():
                try:
                    msa[i, j] = data.acid_to_index(acid)
                except KeyError:
                    msa[i, j] = data.num_acids()
                j += 1
            elif acid == "-":
                msa[i, j] = -1
                j += 1
        assert j == len(sequence)
    return msa
