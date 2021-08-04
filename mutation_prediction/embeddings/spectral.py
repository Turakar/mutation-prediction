import numpy as np
import scipy.fft as fft
from optuna import Trial
from pyfftw.interfaces import scipy_fft as pyfftw

import mutation_prediction.embeddings.aaindex as aaindex
from mutation_prediction.data import Dataset
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import HyperparameterBool, HyperparameterDict


class SpectralSingle(Embedding):
    def __init__(self, key: str = "ZHOH040101"):
        indices = aaindex.read_aaindex1(keys=[key])[:, 0]
        self.indices = indices - np.mean(indices)

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        sequences = dataset.get_sequences()
        encoded = self.indices[sequences]
        n = len(dataset.get_sequence())
        # fy = fft(encoded, axis=1)[:, : n // 2]
        with fft.set_backend(pyfftw):
            fy = fft.rfft(encoded, axis=1)
        magnitude = 2.0 / n * np.abs(fy)
        return magnitude


class Spectral(Embedding):
    def __init__(self, embedding: Embedding):
        self.embedding = embedding

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        return self._spectral(self.embedding.embed_update(dataset))

    def embed(self, dataset: Dataset) -> np.ndarray:
        return self._spectral(self.embedding.embed(dataset))

    def _spectral(self, embedded: np.ndarray) -> np.ndarray:
        n = embedded.shape[1]
        # fy = fft(embedded, axis=1)[:, : n // 2, :]
        with fft.set_backend(pyfftw):
            fy = fft.rfft(embedded, axis=1)
        magnitude = 2.0 / n * np.abs(fy)
        return magnitude


class OptionalSpectral(Spectral):
    def __init__(self, embedding: Embedding):
        super(OptionalSpectral, self).__init__(embedding)
        self.hyperparams = HyperparameterDict({"spectral": HyperparameterBool()})
        if hasattr(embedding, "hyperparams"):
            self.hyperparams["embedding"] = embedding.hyperparams

    def _spectral(self, embedded: np.ndarray) -> np.ndarray:
        if self.hyperparams["spectral"].get():
            return super(OptionalSpectral, self)._spectral(embedded)
        else:
            return embedded
