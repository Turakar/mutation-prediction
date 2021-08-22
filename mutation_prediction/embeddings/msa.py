import os.path
import tempfile
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna import Trial
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import mutation_prediction.data as data
import mutation_prediction.data.preprocessing as preprocessing
import mutation_prediction.node as node
from mutation_prediction.data import Dataset, msa_helper
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import (
    HyperparameterCategorical,
    HyperparameterDict,
    HyperparameterFloat,
    HyperparameterInt,
)
from mutation_prediction.models.lightning import (
    EarlyStoppingMinEpochs,
    TrialUserAttrsCallback,
    UnsupervisedTorchDataset,
)


class MlpVariationalAutoEncoder(Embedding):
    def __init__(self, max_epochs=10000):
        self.hyperparams = HyperparameterDict(
            {
                "validation_fraction": HyperparameterFloat(),
                "num_msas": HyperparameterInt(),
                "batch_size": HyperparameterInt(),
                "hidden": HyperparameterInt(),
                "layers": HyperparameterInt(),
                "latent": HyperparameterInt(),
                "dropout": HyperparameterFloat(),
                "kl_coefficient": HyperparameterFloat(),
                "kl_epoch_full": HyperparameterInt(),
            }
        )
        self.model: Union[None, _MlpVariationalAutoEncoder] = None
        self.max_epochs = max_epochs
        self.normalizer = preprocessing.Normalizer()

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        msa = msa_helper.make_msa(dataset)[: self.hyperparams["num_msas"].get()]
        msa_train, msa_val = train_test_split(
            msa, test_size=self.hyperparams["validation_fraction"].get()
        )
        loader_train = DataLoader(
            UnsupervisedTorchDataset(msa_train, dtype=torch.long),
            batch_size=self.hyperparams["batch_size"].get(),
            shuffle=True,
        )
        loader_val = DataLoader(
            UnsupervisedTorchDataset(msa_val, dtype=torch.long),
            batch_size=self.hyperparams["batch_size"].get(),
            shuffle=False,
        )
        self.model = _MlpVariationalAutoEncoder(
            msa_train.shape[1],
            [self.hyperparams["hidden"].get()] * self.hyperparams["layers"].get(),
            self.hyperparams["latent"].get(),
            self.hyperparams["dropout"].get(),
            self.hyperparams["kl_coefficient"].get(),
            self.hyperparams["kl_epoch_full"].get(),
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss", save_top_k=1, mode="min"
        )
        callbacks = [
            # pl.callbacks.EarlyStopping(monitor="val_loss", patience=200, mode="min"),
            EarlyStoppingMinEpochs(
                monitor="val_loss",
                min_epochs=max(500, self.hyperparams["kl_epoch_full"].get() + 200),
                patience=100,
            ),
            checkpoint_callback,
        ]
        if trial is not None:
            callbacks.append(
                TrialUserAttrsCallback(
                    trial,
                    [
                        "val_loss",
                        "val_loss_reconstruction",
                        "val_accuracy",
                        "loss",
                        "loss_kl_divergence",
                        "loss_reconstruction",
                    ],
                    prefix="vae_",
                    every_n_epochs=100,
                )
            )
        with tempfile.TemporaryDirectory() as temp:
            trainer = pl.Trainer(
                callbacks=callbacks,
                logger=False,
                max_epochs=self.max_epochs,
                gpus=node.get_pl_gpus(),
                weights_summary=None,
                progress_bar_refresh_rate=0,
                default_root_dir=temp,
            )
            trainer.fit(self.model, loader_train, loader_val)
            self.model = self.model.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                sequence_length=msa_train.shape[1],
                hidden=[self.hyperparams["hidden"].get()] * self.hyperparams["layers"].get(),
                latent=self.hyperparams["latent"].get(),
                dropout=self.hyperparams["dropout"].get(),
                kl_coefficient=self.hyperparams["kl_coefficient"].get(),
                kl_epoch_full=self.hyperparams["kl_epoch_full"].get(),
            )

        embedded = self._embed(dataset)
        return self.normalizer.norm_update(embedded)

    def embed(self, dataset: Dataset) -> np.ndarray:
        embedded = self._embed(dataset)
        return self.normalizer.norm(embedded)

    def _embed(self, dataset: Dataset) -> np.ndarray:
        batch_size = 1024
        sequences = dataset.get_sequences()
        embedded = np.zeros(
            (sequences.shape[0], self.hyperparams["latent"].get()), dtype=np.float32
        )
        with torch.no_grad():
            for i in range(0, sequences.shape[0], batch_size):
                sequence_batch = torch.tensor(
                    sequences[i : i + batch_size], dtype=torch.long, device=self.model.device
                )
                embedded_batch, _ = self.model(sequence_batch)
                embedded[i : i + batch_size] = embedded_batch.numpy()
        return embedded


class _MlpVariationalAutoEncoder(pl.LightningModule):
    def __init__(self, sequence_length, hidden, latent, dropout, kl_coefficient, kl_epoch_full):
        super(_MlpVariationalAutoEncoder, self).__init__()
        self.dropout = dropout
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        last_size = sequence_length * (data.num_acids() + 1)
        for units in hidden:
            self.encoder_layers.append(nn.Linear(last_size, units))
            self.decoder_layers.insert(0, nn.Linear(units, last_size))
            last_size = units
        self.encoder_mean = nn.Linear(last_size, latent)
        self.encoder_log_var = nn.Linear(last_size, latent)
        self.decoder_first = nn.Linear(latent, last_size)
        self.kl_coefficient = kl_coefficient
        self.kl_epoch_full = kl_epoch_full

    def forward(self, x):
        # batch x sequence
        out = F.one_hot(x + 1, num_classes=data.num_acids() + 1).float()
        # batch x sequence x 21
        out = torch.reshape(out, (out.shape[0], -1))
        # batch x (sequence * 21)
        for layer in self.encoder_layers:
            out = F.relu(layer(F.dropout(out, p=self.dropout)))
        out = F.dropout(out, p=self.dropout)
        mean = self.encoder_mean(out)
        log_var = self.encoder_log_var(out)
        return mean, log_var

    def decode(self, z):
        out = F.relu(self.decoder_first(z))
        for layer in self.decoder_layers:
            out = F.relu(layer(F.dropout(out, p=self.dropout)))
        # batch x (sequence * 21)
        out = torch.reshape(out, (out.shape[0], -1, data.num_acids() + 1))
        # batch x sequence x 21
        return out

    def step(self, batch, sample=True):
        # encode
        mean, log_var = self(batch)
        # sample
        if sample:
            epsilon = torch.normal(mean=0, std=1, size=log_var.shape, device=mean.device)
            z = mean + torch.exp(log_var / 2) * epsilon
        else:
            z = mean
        # decode
        decoded = self.decode(z)
        # calculate loss
        decoded_class_first = torch.transpose(decoded, 1, 2)
        reconstruction_loss = torch.mean(
            F.cross_entropy(decoded_class_first, batch + 1, reduction="mean")
        )
        kl_divergence = torch.mean(
            -0.5 * torch.sum(1 + log_var - torch.square(mean) - torch.exp(log_var), dim=-1)
        )
        if self.current_epoch >= self.kl_epoch_full:
            kl_coefficient = self.kl_coefficient
        else:
            kl_coefficient = self.current_epoch / self.kl_epoch_full * self.kl_coefficient
        loss = reconstruction_loss + kl_coefficient * kl_divergence
        return kl_divergence, loss, reconstruction_loss, decoded

    def training_step(self, batch, batch_idx):
        kl_divergence, loss, reconstruction_loss, _ = self.step(batch)
        self.log("loss_reconstruction", reconstruction_loss, on_step=False, on_epoch=True)
        self.log("loss_kl_divergence", kl_divergence, on_step=False, on_epoch=True)
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        kl_divergence, loss, reconstruction_loss, decoded = self.step(batch, sample=False)
        self.log("val_loss_reconstruction", reconstruction_loss, on_step=False, on_epoch=True)
        self.log("val_loss_kl_divergence", kl_divergence, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        reconstructed = torch.argmax(decoded, dim=-1) - 1
        correct = torch.count_nonzero(torch.eq(reconstructed, batch)).item()
        accuracy = correct / (batch.shape[0] * batch.shape[1])
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class MsaPca(Embedding):
    def __init__(self):
        self.hyperparams = HyperparameterDict(
            {
                "n_components": HyperparameterInt(),
            }
        )
        self.model: Union[None, PCA] = None
        self.normalizer = preprocessing.Normalizer()

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        msas = preprocessing.preprocess_msa(dataset)

        self.model = PCA(n_components=self.hyperparams["n_components"].get())
        self.model.fit(msas)

        embedded = self.model.transform(dataset.get_sequences())
        return self.normalizer.norm_update(embedded)

    def embed(self, dataset: Dataset) -> np.ndarray:
        embedded = self.model.transform(dataset.get_sequences())
        return self.normalizer.norm(embedded)


class PrecomputedAutoEncoder(Embedding):
    def __init__(self, name: str):
        self.hyperparams = HyperparameterDict(
            {
                "latent": HyperparameterCategorical(),
            }
        )
        self.name = name

    def embed_update(self, dataset: Dataset, trial: Trial = None) -> np.ndarray:
        return self.embed(dataset)

    def embed(self, dataset: Dataset) -> np.ndarray:
        path = os.environ.get(
            "MUTATION_PREDICTION_PATCH_AE",
            os.path.join(
                node.get_precomputed_path(),
                self.name + "-%s-%d.npy" % (dataset.get_name(), self.hyperparams["latent"].get())
            )
        )
        probabilities = np.load(
            path,
            mmap_mode="r",
        )
        return probabilities[dataset.get_ids()]
