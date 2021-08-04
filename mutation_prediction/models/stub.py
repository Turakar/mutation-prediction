import numpy as np
from optuna import Trial

from mutation_prediction.data import Dataset
from mutation_prediction.embeddings.msa import MlpVariationalAutoEncoder
from mutation_prediction.models import HyperparameterDict, SelfScoringModel


class MlpVariationalAutoEncoderStub(SelfScoringModel):
    def __init__(self):
        self.embedding = MlpVariationalAutoEncoder()
        super(MlpVariationalAutoEncoderStub, self).__init__(
            HyperparameterDict({"embedding": self.embedding.hyperparams})
        )

    def fit_and_score(self, dataset: Dataset, trial: Trial = None) -> float:
        self.embedding.embed_update(dataset, trial=trial)
        value = min(trial.user_attrs["log_vae_val_loss"][0])
        return value

    def predict(self, dataset: Dataset) -> np.ndarray:
        raise NotImplementedError()
