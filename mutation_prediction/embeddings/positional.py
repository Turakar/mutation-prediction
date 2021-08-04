import numpy as np
from optuna import Trial

from mutation_prediction.data import Dataset, structure_utils
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import HyperparameterDict, HyperparameterInt


class PositionalEmbedding(Embedding):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.hyperparams = HyperparameterDict({"dimensions": HyperparameterInt()})
        self.omg = None

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        dimensions = self.hyperparams["dimensions"].get()
        assert dimensions % 2 == 0
        assert dimensions >= 4
        n = len(dataset.get_sequence())
        k = np.arange(dimensions / 2)  # until k=d inclusive
        self.omg = np.pi * n ** (-2 * k / (dimensions - 2))
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        n = len(dataset.get_sequence())
        embedding = np.zeros((len(dataset), n, len(self.omg) * 2))
        i = np.arange(n)
        embedding[:, :, 0::2] = np.sin(np.outer(i, self.omg))
        embedding[:, :, 1::2] = np.cos(np.outer(i, self.omg))
        return embedding


class StructurePositionalEmbeddingOld(Embedding):
    def __init__(self, g: float = 2.0):
        super(StructurePositionalEmbeddingOld, self).__init__()
        self.hyperparams = HyperparameterDict(
            {
                "positions": HyperparameterInt(),
            }
        )
        self.omg = None
        self.g = g

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        positions = structure_utils.get_positions(dataset)
        positions = positions - np.nanmean(positions, axis=0)
        elongation = np.nanmax(np.abs(positions))
        d = self.hyperparams["positions"].get()
        k = np.arange(d + 1)
        self.omg = np.pi / elongation * (self.g / elongation) ** (-2 * k / d)
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        positions = structure_utils.get_positions(dataset)
        positions = positions - np.nanmean(positions, axis=0)
        d = self.hyperparams["positions"].get()
        p = np.zeros((len(positions), 3, 2 * d + 2))
        for dim in range(3):
            p[:, dim, 0::2] = np.sin(np.outer(positions[:, dim], self.omg))
            p[:, dim, 1::2] = np.cos(np.outer(positions[:, dim], self.omg))
        p = np.reshape(p, (len(positions), -1))
        embedding = np.broadcast_to(p, (len(dataset),) + p.shape)
        return embedding


class StructurePositionalEmbedding(Embedding):
    def __init__(self, resolution: float = 2.0):
        super(StructurePositionalEmbedding, self).__init__()
        self.hyperparams = HyperparameterDict(
            {
                "frequencies": HyperparameterInt(),
            }
        )
        self.omg = None
        self.resolution = resolution

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        positions = structure_utils.get_positions(dataset)
        positions = positions - np.nanmean(positions, axis=0)
        elongation = np.nanmax(np.abs(positions))
        frequencies = self.hyperparams["frequencies"].get()
        k = np.arange(frequencies + 1)
        self.omg = np.pi / self.resolution * (elongation / self.resolution) ** (-k / frequencies)
        return self.embed(dataset)

    def embed(self, dataset: Dataset):
        positions = structure_utils.get_positions(dataset)
        positions = positions - np.nanmean(positions, axis=0)
        frequencies = self.hyperparams["frequencies"].get()
        p = np.zeros((len(positions), 3, 2 * (frequencies + 1)))
        for dim in range(3):
            p[:, dim, 0::2] = np.sin(np.outer(positions[:, dim], self.omg))
            p[:, dim, 1::2] = np.cos(np.outer(positions[:, dim], self.omg))
        p = p.reshape((len(positions), -1))
        embedding = np.broadcast_to(p, (len(dataset),) + p.shape)
        return embedding
