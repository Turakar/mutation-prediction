import numpy as np
from optuna import Trial

from mutation_prediction.data import Dataset
from mutation_prediction.data import structure_utils as structure_utils
from mutation_prediction.embeddings import Embedding
from mutation_prediction.embeddings.aaindex import read_aaindex3


class SPairs(Embedding):
    matrix = None

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        if SPairs.matrix is None:
            matrix = read_aaindex3()
            SPairs.matrix = (matrix - matrix.mean(axis=(0, 1))) / matrix.std(axis=(0, 1))
        contacts = structure_utils.get_contacts(dataset)
        embedding = np.zeros(
            (len(dataset), len(contacts), SPairs.matrix.shape[2]), dtype=np.float32
        )
        for i, sequence in enumerate(dataset.get_sequences()):
            contact_acids = sequence[contacts.reshape(-1)].reshape(-1, 2)
            embedding[i] = SPairs.matrix[contact_acids[:, 0], contact_acids[:, 1]]
        return embedding
