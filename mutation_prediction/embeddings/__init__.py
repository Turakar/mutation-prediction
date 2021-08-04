import abc

import numpy as np
from optuna import Trial

from mutation_prediction.data import Dataset


class Embedding(abc.ABC):
    @abc.abstractmethod
    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        pass

    @abc.abstractmethod
    def embed(self, dataset: Dataset) -> np.ndarray:
        pass


class EmbeddingMatrix(Embedding):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        return self.matrix[dataset.get_sequences()]

    def get_matrix(self) -> np.ndarray:
        return self.matrix
