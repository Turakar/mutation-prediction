import numpy as np
from optuna import Trial
from sklearn.decomposition import PCA

import mutation_prediction.data as data
import mutation_prediction.embeddings.aaindex as aaindex
from mutation_prediction.data import Dataset
from mutation_prediction.embeddings import Embedding, EmbeddingMatrix


class AcidsOneHot(EmbeddingMatrix):
    def __init__(self):
        super(AcidsOneHot, self).__init__(np.eye(data.num_acids()))


class ZScales(EmbeddingMatrix):
    def __init__(self):
        super(ZScales, self).__init__(
            np.asarray(
                [
                    [0.24, -2.32, 0.60, -0.14, 1.30],
                    [0.84, -1.67, 3.75, 0.18, -2.65],
                    [3.98, 0.93, 1.93, -2.46, 0.75],
                    [3.11, 0.26, -0.11, -3.04, -0.25],
                    [-4.22, 1.94, 1.06, 0.54, -0.62],
                    [2.05, -4.06, 0.36, -0.82, -0.38],
                    [2.47, 1.95, 0.26, 3.90, 0.09],
                    [-3.89, -1.73, -1.71, -0.84, 0.26],
                    [2.29, 0.89, -2.49, 1.49, 0.31],
                    [-4.28, -1.30, -1.49, -0.72, 0.84],
                    [-2.85, -0.22, 0.47, 1.94, -0.98],
                    [3.05, 1.62, 1.04, -1.15, 1.61],
                    [1.66, 0.27, 1.84, 0.70, 2.00],
                    [1.75, 0.50, -1.44, -1.34, 0.66],
                    [3.52, 2.50, -3.50, 1.99, -0.17],
                    [2.39, -1.07, 1.15, -1.39, 0.67],
                    [0.75, -2.18, -1.12, -1.46, -0.40],
                    [-2.59, -2.64, -1.54, -0.85, -0.02],
                    [-4.36, 3.94, 0.59, 3.44, -1.59],
                    [-2.54, 2.44, 0.43, 0.04, -1.47],
                ]
            )
        )


class VHSE(EmbeddingMatrix):
    def __init__(self):
        super(VHSE, self).__init__(
            np.asarray(
                [
                    [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
                    [0.18, -1.67, -0.46, -0.21, 0.0, 1.2, -1.61, -0.19],
                    [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
                    [-1.18, 0.4, 0.1, 0.36, -2.16, -0.17, 0.91, 0.02],
                    [1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.2],
                    [-0.2, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
                    [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
                    [1.27, -0.14, 0.3, -1.8, 0.3, -1.61, -0.16, -0.13],
                    [-1.17, 0.7, 0.7, 0.8, 1.64, 0.67, 1.63, 0.13],
                    [1.36, 0.07, 0.26, -0.8, 0.22, -1.37, 0.08, -0.62],
                    [1.01, -0.53, 0.43, 0.0, 0.23, 0.1, -0.86, -0.68],
                    [-0.99, 0.0, -0.37, 0.69, -0.55, 0.85, 0.73, -0.8],
                    [0.22, -0.17, -0.5, 0.05, -0.01, -1.34, -0.19, 3.56],
                    [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.2, -0.41],
                    [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.3, 0.83],
                    [-0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
                    [-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39],
                    [0.76, -0.92, -0.17, -1.91, 0.22, -1.4, -0.24, -0.03],
                    [1.5, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85],
                    [0.61, 1.6, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52],
                ]
            )
        )


class PcScores(EmbeddingMatrix):
    matrix = None

    def __init__(self):
        if PcScores.matrix is None:
            matrix = aaindex.read_aaindex1()
            matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
            pca = PCA(n_components=11)
            PcScores.matrix = pca.fit_transform(matrix)
        super(PcScores, self).__init__(PcScores.matrix)


class SScales(EmbeddingMatrix):
    matrix = None

    def __init__(self):
        if SScales.matrix is None:
            matrix = aaindex.read_aaindex1(
                keys=[
                    "BIOV880101",
                    "BLAM930101",
                    "NAGK730101",
                    "TSAJ990101",
                    "NAKH920106",
                    "NAKH920107",
                    "NAKH920108",
                    "CEDJ970104",
                    "LIFS790101",
                    "MIYS990104",
                    "ARGP820101",
                    "DAWD720101",
                    "FAUJ880109",
                ]
            )
            SScales.matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        super(SScales, self).__init__(SScales.matrix)


class AaIndex(EmbeddingMatrix):
    matrix = None

    def __init__(self, keys=None):
        if AaIndex.matrix is None:
            matrix = aaindex.read_aaindex1(keys=keys)
            AaIndex.matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        super(AaIndex, self).__init__(AaIndex.matrix)


class ProtVec(Embedding):
    matrix = None

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        if ProtVec.matrix is None:
            ProtVec.matrix = _read_protvec()
        embedded = np.zeros((len(dataset), 100))
        sequence_length = len(dataset.get_sequence())
        num_mutations = dataset.get_num_mutations()
        positions = dataset.get_positions()
        for i, sequence in enumerate(dataset.get_sequences()):
            three_grams = set()
            for p in positions[i, : num_mutations[i]]:
                p = int(p)
                if 2 <= p:
                    three_grams.add((p - 2, p - 1, p))
                if 1 <= p < sequence_length - 1:
                    three_grams.add((p - 1, p, p + 1))
                if p < sequence_length - 2:
                    three_grams.add((p, p + 1, p + 2))
            vectors = [
                ProtVec.matrix[sequence[g[0]], sequence[g[1]], sequence[g[2]]] for g in three_grams
            ]
            if len(vectors) > 0:
                embedded[i] = np.asarray(vectors).mean(axis=0)
        return embedded

    def get_matrix(self) -> np.ndarray:
        return self.matrix


def _read_protvec():
    with open("data/embeddings/protVec_100d_3grams.csv") as fd:
        matrix = np.zeros((data.num_acids(), data.num_acids(), data.num_acids(), 100))
        for row in fd:
            row = row.replace('"', "")
            parts = row.split("\t")
            try:
                i = data.acid_to_index(parts[0][0])
                j = data.acid_to_index(parts[0][1])
                k = data.acid_to_index(parts[0][2])
            except KeyError:
                continue
            vector = np.asarray([float(v) for v in parts[1:]])
            matrix[i, j, k] = vector
        return matrix
