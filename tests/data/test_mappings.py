import io

import numpy as np

import mutation_prediction.data as data


def test_acid_shortname_to_letter_examples():
    assert data.acid_shortname_to_letter("Ala") == "A"
    assert data.acid_shortname_to_letter("Leu") == "L"
    assert data.acid_shortname_to_letter("Lys") == "K"


def test_genome_to_acid_examples():
    assert data.genome_to_acid("TCG") == "S"
    assert data.genome_to_acid("ACG") == "T"
    assert data.genome_to_acid("TGG") == "W"


def test_genome_to_acid_plausibility():
    genes = ["T", "C", "A", "G"]
    found = set()
    for i in range(data.num_genes()):
        for j in range(data.num_genes()):
            for k in range(data.num_genes()):
                genome = genes[i] + genes[j] + genes[k]
                if genome not in ["TAA", "TAG", "TGA"]:
                    found.add(data.genome_to_acid(genome))
    assert len(found) == data.num_acids()
    for acid in found:
        assert isinstance(data.acid_to_index(acid), int)


def test_acid_index_mapping():
    known = set()
    for i in range(data.num_acids()):
        acid = data.index_to_acid(i)
        assert acid not in known
        known.add(acid)
    for acid in known:
        assert acid == data.index_to_acid(data.acid_to_index(acid))


def test_sequence_genome_to_acid():
    genome_sequence = "ATACACGAT"
    acid_sequence = (
        data.acid_shortname_to_letter("Ile")
        + data.acid_shortname_to_letter("His")
        + data.acid_shortname_to_letter("Asp")
    )
    assert data.sequence_genome_to_acid(genome_sequence) == acid_sequence


def test_sequence_to_string():
    acid_sequence = "MLKFPTQ"
    sequence = np.asarray([data.acid_to_index(a) for a in acid_sequence])
    assert data.sequence_to_string(sequence) == acid_sequence


def test_write_to_fasta():
    sequences = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
    s = io.StringIO()
    data.write_to_fasta(s, sequences, id="C%d")
    out = s.getvalue()
    assert out == ">C0 <unknown description>\nACDE\n>C1 <unknown description>\nEDCA\n"


def test_ids():
    dataset = data.Dataset(
        np.zeros(10),
        np.zeros((5, 0)),
        np.zeros((5, 0)),
        np.zeros(5),
    )
    assert list(dataset.get_ids()) == [0, 1, 2, 3, 4]
    sub1 = dataset[((0, 3, 4),)]
    assert list(sub1.get_ids()) == [0, 3, 4]
    sub2 = dataset[((1,),)]
    assert list(sub2.get_ids()) == [1]
