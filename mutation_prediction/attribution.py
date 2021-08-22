import gc
import os
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import optuna.storages
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from glmnet import ElasticNet
from numpy.lib.format import open_memmap
from sklearn.svm import SVR
from tqdm.auto import tqdm
from transformers import (
    AlbertTokenizer,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    T5EncoderModel,
    T5Tokenizer,
    XLNetTokenizer,
)

from mutation_prediction import data
from mutation_prediction.cli import registry
from mutation_prediction.data import Dataset
from mutation_prediction.embeddings import Embedding
from mutation_prediction.embeddings.msa import PrecomputedAutoEncoder
from mutation_prediction.embeddings.other import ConcatEmbedding, Linearize
from mutation_prediction.embeddings.prottrans import (
    ProtTransFeatureExtractionPrecomputed,
)
from mutation_prediction.embeddings.spectral import OptionalSpectral, Spectral
from mutation_prediction.models import Model
from mutation_prediction.models.cnn import KCnn
from mutation_prediction.models.mlp import MlpRegressor
from mutation_prediction.models.msa import LinearAutoEncoder


class TorchGlmnet(nn.Module):
    def __init__(self, glmnet: ElasticNet):
        super(TorchGlmnet, self).__init__()
        coefficients = glmnet.coef_path_[:, 0]
        mask = coefficients != 0
        self.indices = nn.Parameter(torch.tensor(np.nonzero(mask)[0]), requires_grad=False)
        self.coefficients = nn.Parameter(
            torch.tensor(coefficients[mask]).unsqueeze(0), requires_grad=False
        )
        self.intercept = nn.Parameter(torch.tensor(glmnet.intercept_path_[0]), requires_grad=False)

    def forward(self, x: torch.Tensor):
        return self.intercept + torch.sum(self.coefficients * x[:, self.indices], dim=1)


class TorchEpsilonSvr(nn.Module):
    def __init__(self, svr: SVR):
        super(TorchEpsilonSvr, self).__init__()
        self.coefficients = nn.Parameter(torch.tensor(svr.dual_coef_[0]).unsqueeze(0))
        self.support_vectors = nn.Parameter(torch.tensor(svr.support_vectors_))
        self.intercept = nn.Parameter(torch.tensor(svr.intercept_))
        self.gamma = float(svr.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.exp(
            -self.gamma * torch.norm(x.unsqueeze(1) - self.support_vectors.unsqueeze(0), dim=2) ** 2
        )
        out = torch.sum(self.coefficients * k, dim=1) + self.intercept
        return out


class TorchAutoencoder(nn.Module):
    def __init__(self, model: LinearAutoEncoder):
        super(TorchAutoencoder, self).__init__()
        self.encoder = model.model.encoder
        self.decoder = model.model.decoder

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        z = self.encoder(x)
        logits = self.decoder(z)
        logits = torch.reshape(logits, (logits.shape[0], -1, data.num_acids() + 2))
        return F.softmax(logits, dim=-1)


class TorchKCnn(nn.Module):
    def __init__(self, model: KCnn):
        super(TorchKCnn, self).__init__()
        model = model.model.cpu()
        self.filters = model.filters
        self.dense = model.dense
        self.neighborhood = model.neighborhood
        self.distances = model.distances

    def forward(self, x, phases=None):
        if phases is None:
            phases = {"input", "filter", "max", "final"}
        if "input" in phases:
            x = self.make_k_tensor(x)
        if "filter" in phases:
            batch_size = x.shape[0]
            sequence_len = x.shape[1]
            x = torch.reshape(x, (batch_size * sequence_len, -1))
            x = self.filters(x)
            x = torch.reshape(x, (batch_size, sequence_len, -1))
        if "max" in phases:
            x = torch.max(x, dim=1)[0]
        if "final" in phases:
            x = F.relu(x)
            x = self.dense(x)
            x = x.squeeze(1)
        return x

    def make_k_tensor(self, x):
        distances = self.distances.unsqueeze(0).expand(x.shape[0], -1, -1).unsqueeze(3)
        x = torch.cat([x[:, self.neighborhood, :], distances], dim=-1)
        return x


class TorchMlp(nn.Module):
    def __init__(self, model: MlpRegressor):
        super(TorchMlp, self).__init__()
        model = model.model.cpu()
        self.layers = model.layers

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class TorchProtTrans(nn.Module):
    def __init__(self, embedding: ProtTransFeatureExtractionPrecomputed, half: bool = False):
        super(TorchProtTrans, self).__init__()
        self.name = "Rostlab/" + embedding.hyperparams["model"].get()
        tokenizer_generator = AutoTokenizer
        model_factory = AutoModel
        if "t5" in self.name:
            tokenizer_generator = T5Tokenizer
            model_factory = T5EncoderModel
        elif "xlnet" in self.name:
            tokenizer_generator = XLNetTokenizer
        elif "albert" in self.name:
            tokenizer_generator = AlbertTokenizer
        self.tokenizer = tokenizer_generator.from_pretrained(self.name, do_lower_case=False)
        self.model: PreTrainedModel = model_factory.from_pretrained(self.name)
        gc.collect()
        if half:
            self.model = self.model.half()
        self.norm_mean = nn.Parameter(torch.tensor(embedding.normalizer.mean[0], dtype=torch.float))
        self.norm_max = nn.Parameter(
            torch.tensor(embedding.normalizer.maximum[0], dtype=torch.float)
        )

    def forward(self, input_ids, attention_mask, seq_len):
        embedding = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = self.reduce_to_sequence(embedding, seq_len)
        embedding = (embedding - self.norm_mean) / self.norm_max
        return embedding

    def reduce_to_sequence(self, embedding, seq_len):
        if "electra" in self.name or "bert" in self.name or "electra" in self.name:
            embedding = embedding[:, 1 : 1 + seq_len]
        elif "xlnet" in self.name:
            padded_seq_len = len(embedding)
            embedding = embedding[:, padded_seq_len - seq_len - 2 : padded_seq_len - 2]
        elif "t5" in self.name:
            embedding = embedding[:, :seq_len]
        else:
            raise ValueError("Unknown model name! Cannot determine correct indexing.")
        return embedding

    def tokenize(self, sequences_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            " ".join(data.sequence_to_string(s.detach().cpu().numpy())) for s in sequences_tensor
        ]
        ids = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
        return torch.tensor(ids["input_ids"], device=sequences_tensor.device), torch.tensor(
            ids["attention_mask"], device=sequences_tensor.device
        )


class TorchPipeline(nn.Module):
    def __init__(self, embedding: Embedding, submodule: nn.Module, half_prottrans: bool = False):
        super(TorchPipeline, self).__init__()
        self.embedding = embedding
        self.submodule = submodule
        self.autoencoder: Optional[TorchAutoencoder] = None
        self.prottrans: Optional[TorchProtTrans] = None
        self.non_prottrans = nn.Identity()
        self.half_prottrans = half_prottrans

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[List[Tuple[bool, bool, int]]] = None,
        *sub_args: Any
    ) -> torch.Tensor:
        if indices is not None:
            mask = torch.zeros(x.shape[2], dtype=torch.bool)
            index = 0
            for _, _, pt, length in indices:
                if not pt:
                    mask[index : index + length] = True
                index += length
            x[:, :, mask] = self.non_prottrans(x[:, :, mask])
            parts = []
            index = 0
            for spectral, ae, pt, length in indices:
                y = x[:, :, index : index + length]
                if ae:
                    y = self.autoencoder(y)
                elif pt:
                    input_ids, attention_mask = self.prottrans.tokenize(y[..., 0].long())
                    y = self.prottrans(input_ids, attention_mask, x.shape[1]).float()
                if not spectral:
                    parts.append(y.reshape(x.shape[0], -1))
                else:
                    n = y.shape[1]
                    fy = torch.fft.rfft(y, dim=1)
                    mag = 2.0 / n * torch.abs(fy)
                    parts.append(mag.reshape(mag.shape[0], -1))
                index += length
            x = torch.cat(parts, dim=1)
        return self.submodule(x, *sub_args)

    def embed(
        self,
        dataset: Dataset,
        spectral: bool = False,
        autoencoder: bool = True,
        prottrans: bool = True,
        storage: Optional[optuna.storages.BaseStorage] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[bool, bool, bool, int]]]]:
        if autoencoder and spectral:
            x = torch.tensor(self.embedding.embed(dataset), dtype=torch.float)
            return x, None
        else:
            if isinstance(self.embedding, ConcatEmbedding):
                embeddings = self.embedding.get_selected_embeddings()
            else:
                embeddings = [self.embedding]

            def handle_embedding(embedding):

                if not autoencoder and isinstance(embedding, PrecomputedAutoEncoder):
                    if self.autoencoder is None:
                        print("Preparing autoencoder...")
                        assert embedding.name == "Linear"
                        fmt_args = (dataset.get_name(), embedding.hyperparams["latent"].get())
                        study_name = "testLinearAutoEncoder_%sfull_%d" % fmt_args
                        parameters = os.path.join("models", "Linear-%s-%d" % fmt_args)
                        study = optuna.load_study(study_name, storage)
                        ae = LinearAutoEncoder()
                        ae.hyperparams.set_from_trial(study.best_trial, study.user_attrs["params"])
                        ae_args = ae._get_model_args(
                            len(registry.datasets[study.user_attrs["dataset"]]()[0].get_sequence())
                        )
                        ae.model = ae.model_cls(**ae_args)
                        ae.model.load_state_dict(torch.load(parameters))
                        self.autoencoder = TorchAutoencoder(ae)
                    return True, False, np.eye(22)[dataset.get_sequences() + 1]

                elif not prottrans and isinstance(embedding, ProtTransFeatureExtractionPrecomputed):
                    if self.prottrans is None:
                        print("Preparing ProtTrans...")
                        self.prottrans = TorchProtTrans(embedding, half=self.half_prottrans)
                    return False, True, dataset.get_sequences()[..., None]

                else:
                    return False, False, embedding.embed(dataset)

            parts = []
            for embedding in embeddings:
                assert isinstance(embedding, Linearize)
                non_linear = embedding.embedding
                if (
                    isinstance(non_linear, OptionalSpectral)
                    and non_linear.hyperparams["spectral"].get()
                ) or type(non_linear) is Spectral:
                    ae, pt, embedded = handle_embedding(non_linear.embedding)
                    parts.append((True, ae, pt, embedded))
                elif isinstance(non_linear, OptionalSpectral):
                    ae, pt, embedded = handle_embedding(non_linear.embedding)
                    parts.append((False, ae, pt, embedded))
                else:
                    ae, pt, embedded = handle_embedding(non_linear)
                    parts.append((False, ae, pt, embedded))

            tensor = torch.tensor(
                np.concatenate(
                    [x for _, _, _, x in parts],
                    axis=2,
                ),
                dtype=torch.float,
            )
            indices = []
            for s, ae, pt, x in parts:
                length = x.shape[2]
                indices.append((s, ae, pt, length))
            return tensor, indices


def attribute(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    ext_batch_size: int = 64,
    int_batch_size: int = 64,
    n_steps: int = 50,
    progress_bar: bool = True,
    progress_bar_leave: bool = True,
    mmap_file: Optional[Union[str, List[str]]] = None,
    max_error: float = 1e-6,
    forward_args: Optional[Any] = None,
    device: Optional[torch.device] = None,
    preprocessing: Callable[[torch.Tensor], torch.Tensor] = None,
    layers: Optional[List[nn.Module]] = None,
) -> Union[np.ndarray, List[np.ndarray]]:
    if device is None:
        device = torch.device("cpu")
    if mmap_file is not None and not isinstance(mmap_file, list):
        mmap_file = [mmap_file]

    # prepare ext. batching
    batch_starts = range(0, x.shape[0], ext_batch_size)
    if progress_bar:
        batch_starts = tqdm(batch_starts, leave=progress_bar_leave)

    # prepare algorithm
    model = model.to(device)
    if layers is None:
        algorithm = IntegratedGradients(model)
    elif len(layers) == 1:
        algorithm = LayerIntegratedGradients(model, layers[0])
    else:
        algorithm = LayerIntegratedGradients(model, layers)
    attributions = None
    baseline = baseline.to(device)
    if preprocessing:
        with torch.no_grad():
            baseline = preprocessing(baseline)

    # iterate over ext. batches
    for batch_start in batch_starts:

        # extract batch
        batch_end = min(x.shape[0], batch_start + ext_batch_size)
        batch = x[batch_start:batch_end].to(device)
        if preprocessing:
            with torch.no_grad():
                batch = preprocessing(batch)

        # run algorithm
        attr, err = algorithm.attribute(
            batch,
            baseline,
            internal_batch_size=int_batch_size,
            return_convergence_delta=True,
            n_steps=n_steps,
            additional_forward_args=forward_args,
        )
        assert torch.max(err) <= max_error, "Error above maximum! %.3E" % err
        if not isinstance(attr, list):
            attr = [attr]
        attr = [a.cpu().numpy() for a in attr]

        # save results
        if attributions is None:
            attributions = []
            for i, sub_attr in enumerate(attr):
                attributions_shape = (x.shape[0],) + sub_attr.shape[1:]
                if mmap_file is None:
                    attributions.append(np.zeros(attributions_shape, dtype=sub_attr.dtype))
                else:
                    attributions.append(
                        open_memmap(
                            mmap_file[i],
                            mode="w+",
                            dtype=sub_attr.dtype,
                            shape=attributions_shape,
                        )
                    )
        for i, sub_attr in enumerate(attr):
            attributions[i][batch_start:batch_end] = sub_attr

    if len(attributions) == 1:
        return attributions[0]
    else:
        return attributions


def extract_wild_type(dataset: Dataset) -> Dataset:
    mask = dataset.get_num_mutations() == 0
    assert np.count_nonzero(mask) == 1
    return dataset[mask]
