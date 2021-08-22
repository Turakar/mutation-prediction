import os
import pickle
from typing import Dict, Union

import numpy as np
from optuna import Trial
from transformers import BertForMaskedLM, BertTokenizer, Pipeline, pipeline

import mutation_prediction.data as data
from mutation_prediction import node
from mutation_prediction.data import Dataset, preprocessing
from mutation_prediction.data.preprocessing import Normalizer
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import HyperparameterCategorical, HyperparameterDict


class ProtBertFeatureExtraction(Embedding):
    def __init__(self):
        self.hyperparams = HyperparameterDict({"selection": HyperparameterCategorical()})
        self.pipeline: Union[None, Pipeline] = None
        self.normalizer = Normalizer()

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
        model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
        self.pipeline = pipeline("feature-extraction", tokenizer=tokenizer, model=model, device=0)
        embeddings = self._embed(dataset)
        return self.normalizer.norm_update(embeddings)

    def embed(self, dataset: Dataset) -> np.ndarray:
        embeddings = self._embed(dataset)
        return self.normalizer.norm(embeddings)

    def _embed(self, dataset: Dataset) -> np.ndarray:
        sequences = [" ".join(data.sequence_to_string(s)) for s in dataset.get_sequences()]
        batch_size = 256
        embeddings = []
        for batch_start in range(0, len(sequences), batch_size):
            batch_end = min(len(sequences), batch_start + batch_size)
            embeddings = embeddings + [
                embedding for embedding in self.pipeline(sequences[batch_start:batch_end])
            ]
        embeddings = np.asarray(embeddings)
        selection = self.hyperparams["selection"].get()
        if selection == "CLS":
            embeddings = embeddings[:, 0, :]
        elif selection == "noCLS":
            embeddings = embeddings[:, 1:-1, :]
        elif selection == "all":
            embeddings = embeddings[:, 0:-1, :]
        else:
            raise ValueError("Unknown selection! %s" % selection)
        return embeddings


class ProtBertFeatureExtractionPrecomputed(Embedding):
    def __init__(self):
        self.hyperparams = HyperparameterDict({"selection": HyperparameterCategorical()})
        self.embeddings: Union[None, Dict[frozenset, np.ndarray]] = None

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        with open(
            os.path.join(node.get_precomputed_path(), "%s.pkl" % dataset.get_name()), "rb"
        ) as index_fd:
            index = pickle.load(index_fd)
        embeddings = np.load(
            os.path.join(node.get_precomputed_path(), "ProtBert-%s.npy" % dataset.get_name())
        )
        self.embeddings = {k: embeddings[i] for k, i in index.items()}
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        tuples = preprocessing.dataset_to_tuples(dataset)
        embedded = np.zeros(
            (len(tuples),) + next(iter(self.embeddings.values())).shape, dtype=np.float32
        )
        for i, mutant in enumerate(tuples):
            key = frozenset(mutant)
            embedded[i] = self.embeddings[key]
        selection = self.hyperparams["selection"].get()
        if selection == "CLS":
            embeddings = embedded[:, 0, :]
        elif selection == "noCLS":
            embeddings = embedded[:, 1:, :]
        elif selection == "all":
            embeddings = embedded[:, :, :]
        else:
            raise ValueError("Unknown selection! %s" % selection)
        return embeddings


class ProtTransFeatureExtractionPrecomputed(Embedding):
    def __init__(self):
        self.hyperparams = HyperparameterDict(
            {
                "model": HyperparameterCategorical(),
            }
        )
        self.normalizer = Normalizer()
        self.sub_dataset = node.get_precomputed_dataset()

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        embedded = self._embed(dataset)
        return self.normalizer.norm_update(embedded)

    def embed(self, dataset: Dataset) -> np.ndarray:
        embedded = self._embed(dataset)
        return self.normalizer.norm(embedded)

    def _embed(self, dataset: Dataset) -> np.ndarray:
        if self.sub_dataset is not None:
            dataset_name = self.sub_dataset
            ids = np.load(
                os.path.join(
                    node.get_precomputed_path(),
                    "%s-%s-ids.npy" % (self.hyperparams["model"].get(), dataset_name),
                )
            )
            mapping = {id: index for index, id in enumerate(ids)}
            indices = np.zeros_like(dataset.get_ids())
            for i, id in enumerate(dataset.get_ids()):
                indices[i] = mapping[id]
        else:
            dataset_name = dataset.get_name()
            indices = dataset.get_ids()
        embeddings = np.load(
            os.path.join(
                node.get_precomputed_path(),
                "%s-%s.npy" % (self.hyperparams["model"].get(), dataset_name),
            ),
            mmap_mode="r",
        )
        return embeddings[indices]
