from typing import List

import numpy as np
from optuna import Trial

from mutation_prediction.data import Dataset
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import HyperparameterDict, HyperparameterOptional


class MutInd(Embedding):
    def __init__(self):
        self.mutation_to_index = {}

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        mutations = set()
        for i, n in enumerate(dataset.get_num_mutations()):
            for p, a in zip(dataset.get_positions()[i, :n], dataset.get_acids()[i, :n]):
                mutations.add((p, a))
        self.mutation_to_index = {}
        for i, m in enumerate(mutations):
            self.mutation_to_index[m] = i
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        embedding = np.zeros((len(dataset), len(self.mutation_to_index)), dtype=np.float32)
        for i, n in enumerate(dataset.get_num_mutations()):
            for p, a in zip(dataset.get_positions()[i, :n], dataset.get_acids()[i, :n]):
                if (p, a) in self.mutation_to_index:
                    embedding[i, self.mutation_to_index[(p, a)]] = 1
        return embedding


class ConcatEmbedding(Embedding):
    def __init__(self, *embeddings: Embedding, configurable: bool = False):
        if configurable:
            hyperparams_dict = {
                "i%d"
                % i: HyperparameterOptional(
                    getattr(embedding, "hyperparams", HyperparameterDict({}))
                )
                for i, embedding in enumerate(embeddings)
            }
        else:
            hyperparams = []
            for i, embedding in enumerate(embeddings):
                if hasattr(embedding, "hyperparams"):
                    hyperparams.append((i, embedding.hyperparams))
            hyperparams_dict = {"i%d" % i: params for (i, params) in hyperparams}
        self.hyperparams = HyperparameterDict(hyperparams_dict)
        self.embeddings = list(embeddings)
        self.configurable = configurable

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        embedded = [
            embedder.embed_update(dataset, trial=trial)
            for embedder in self.get_selected_embeddings()
        ]
        return self._concat(dataset, embedded)

    def embed(self, dataset: Dataset) -> np.ndarray:
        embedded = [embedder.embed(dataset) for embedder in self.get_selected_embeddings()]
        return self._concat(dataset, embedded)

    def get_selected_embeddings(self) -> List[Embedding]:
        if self.configurable:
            return [
                embedder
                for i, embedder in enumerate(self.embeddings)
                if self.hyperparams["i%d" % i].get() is not None
            ]
        else:
            return self.embeddings

    def _concat(self, dataset: Dataset, embedded: List[np.ndarray]) -> np.ndarray:
        if len(embedded) == 0:
            return np.zeros((len(dataset), len(dataset.get_sequence()), 0), dtype=np.float32)
        elif len(embedded) == 1:
            return embedded[0]
        else:
            return np.concatenate(embedded, axis=-1)


class Linearize(Embedding):
    def __init__(self, embedding: Embedding):
        self.embedding = embedding
        if hasattr(embedding, "hyperparams"):
            self.hyperparams = embedding.hyperparams

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        embedded = self.embedding.embed_update(dataset, trial=trial)
        return embedded.reshape((embedded.shape[0], -1))

    def embed(self, dataset: Dataset) -> np.ndarray:
        embedded = self.embedding.embed(dataset)
        return embedded.reshape((embedded.shape[0], -1))
