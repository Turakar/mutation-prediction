import abc
import tempfile
from typing import Any, Dict, List, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from optuna import Trial
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader

from mutation_prediction import data, node
from mutation_prediction.data import Dataset, msa_helper
from mutation_prediction.models import (
    HyperparameterDict,
    HyperparameterFloat,
    HyperparameterInt,
    SelfScoringModel,
)
from mutation_prediction.models.lightning import (
    EarlyStoppingMinEpochs,
    TrialUserAttrsCallback,
)


class SequenceProbabilityModel(SelfScoringModel):
    def __init__(
        self,
        hyperparams: HyperparameterDict,
        model_cls: Type[pl.LightningModule],
        performance_metric: str,
        log_metrics: List[str],
        patience: int = 100,
        max_epochs: int = 10000,
    ):
        super(SequenceProbabilityModel, self).__init__(
            HyperparameterDict(
                {
                    "model": hyperparams,
                    "splits": HyperparameterInt(),
                    "validation_fraction": HyperparameterFloat(),
                    "batch_size": HyperparameterInt(),
                }
            )
        )
        self.model: Union[None, pl.LightningModule] = None
        self.model_cls = model_cls
        self.performance_metric = performance_metric
        self.log_metrics = log_metrics
        self.patience = patience
        self.max_epochs = max_epochs

    def fit(self, dataset: Dataset, trial: Trial = None):
        msa = msa_helper.make_msa(dataset)
        np.random.default_rng().shuffle(msa)
        self._fit(msa, trial=trial)

    def fit_and_score(self, dataset: Dataset, trial: Trial = None) -> float:
        scores = []
        msa = msa_helper.make_msa(dataset)
        np.random.default_rng().shuffle(msa)
        # for i in range(self.hyperparams["iterations"].get()):
        #    train, test = train_test_split(msa, test_size=self.hyperparams["test_fraction"].get())
        for train_idx, val_idx in KFold(n_splits=self.hyperparams["splits"].get()).split(msa):
            train = msa[train_idx]
            val = msa[val_idx]
            self._fit(train, trial=trial)
            scores.append(self._score(val))
            if trial is not None:
                trial.set_user_attr("scores", scores)
        return float(np.mean(scores))

    def _fit(self, msa: np.ndarray, trial: Trial = None):
        train, val = train_test_split(msa, test_size=self.hyperparams["validation_fraction"].get())
        train_dataloader = DataLoader(
            MsaDataset(train),
            batch_size=self.hyperparams["batch_size"].get(),
            shuffle=True,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            MsaDataset(val), batch_size=self.hyperparams["batch_size"].get(), pin_memory=True
        )
        model_params = self._get_model_args(train.shape[1])
        self.model = self.model_cls(**model_params)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=self.performance_metric, save_top_k=1, mode="min"
        )
        callbacks = [
            EarlyStoppingMinEpochs(
                monitor=self.performance_metric,
                min_epochs=self._get_min_epochs(),
                patience=self.patience,
            ),
            checkpoint_callback,
        ]
        if trial is not None:
            callbacks.append(TrialUserAttrsCallback(trial, self.log_metrics, every_n_epochs=100))
        with tempfile.TemporaryDirectory() as temp:
            trainer = pl.Trainer(
                callbacks=callbacks,
                max_epochs=self.max_epochs,
                gpus=node.get_pl_gpus(),
                default_root_dir=temp,
                progress_bar_refresh_rate=0,
                logger=False,
            )
            trainer.fit(self.model, train_dataloader, val_dataloader)
            self.model = self.model.load_from_checkpoint(
                checkpoint_callback.best_model_path, **model_params
            )

    def _score(self, msa: np.ndarray) -> float:
        with torch.no_grad():
            self.model.eval()
            dataloader = DataLoader(
                MsaDataset(msa), batch_size=self.hyperparams["batch_size"].get(), pin_memory=True
            )
            cross_entropies = []
            for i, batch in enumerate(dataloader):
                logits = self.model(batch)
                class_first_logits = torch.transpose(logits, 1, 2)
                cross_entropy = torch.mean(
                    F.cross_entropy(class_first_logits, batch + 1, reduction="mean")
                )
                cross_entropies.append(cross_entropy.item())
        return float(np.mean(cross_entropies))

    def predict(self, dataset: Dataset) -> np.ndarray:
        msa = msa_helper.make_msa(dataset)
        return self.predict_sequences(msa)

    def predict_sequences(self, sequences: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            predictions = np.zeros(sequences.shape + (data.num_acids() + 2,), dtype=np.float32)
            self.model.eval()
            dataloader = DataLoader(
                MsaDataset(sequences),
                batch_size=self.hyperparams["batch_size"].get(),
                pin_memory=True,
            )
            index = 0
            for i, batch in enumerate(dataloader):
                logits = self.model(batch)
                prob = torch.softmax(logits, dim=-1)
                prediction = prob.cpu().numpy()
                length = len(prediction)
                predictions[index : index + length] = prediction
                index += length
            assert index == len(predictions)
        return predictions

    @abc.abstractmethod
    def _get_model_args(self, sequence_length: int) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def _get_min_epochs(self) -> int:
        pass


class MsaDataset(torch.utils.data.Dataset):
    def __init__(self, msa: np.ndarray):
        self.msa = msa

    def __len__(self):
        return len(self.msa)

    def __getitem__(self, item):
        return torch.tensor(self.msa[item], dtype=torch.long)


class MlpAutoEncoder(SequenceProbabilityModel):
    def __init__(self):
        super(MlpAutoEncoder, self).__init__(
            HyperparameterDict(
                {
                    "layers": HyperparameterInt(),
                    "units": HyperparameterInt(),
                    "latent": HyperparameterInt(),
                    "dropout": HyperparameterFloat(),
                    "learning_rate": HyperparameterFloat(),
                }
            ),
            _MlpAutoEncoder,
            "val_loss",
            ["loss", "val_loss", "val_accuracy"],
        )

    def _get_model_args(self, sequence_length: int) -> Dict[str, Any]:
        model_hyperparams = self.hyperparams["model"]
        return dict(
            sequence_length=sequence_length,
            units=[model_hyperparams["units"].get()] * model_hyperparams["layers"].get(),
            latent=model_hyperparams["latent"].get(),
            dropout=model_hyperparams["dropout"].get(),
            learning_rate=model_hyperparams["learning_rate"].get(),
        )

    def _get_min_epochs(self) -> int:
        return 300


class _MlpAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        sequence_length: int,
        units: List[int],
        latent: int,
        dropout: float,
        learning_rate: float,
    ):
        super(_MlpAutoEncoder, self).__init__()
        last_size = sequence_length * (data.num_acids() + 2)
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for u in units:
            self.encoder_layers.append(nn.Linear(last_size, u))
            self.decoder_layers.insert(0, nn.Linear(u, last_size))
            last_size = u
        self.final_encoder = nn.Linear(last_size, latent)
        self.first_decoder = nn.Linear(latent, last_size)
        self.dropout = dropout
        self.learning_rate = learning_rate

    def encode(self, x):
        # batch x sequence
        x = F.one_hot(x + 1, num_classes=data.num_acids() + 2).to(torch.float)
        # batch x sequence x acids
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1))
        # batch x sequence * acids
        for encoder in self.encoder_layers:
            x = F.relu(encoder(F.dropout(x, self.dropout)))
        x = self.final_encoder(F.dropout(x))
        # batch x latent
        return x

    def decode(self, x):
        # batch x latent
        x = F.relu(self.first_decoder(x))
        for decoder in self.decoder_layers:
            x = F.relu(decoder(F.dropout(x, self.dropout)))
        # batch x sequence * acids
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1, data.num_acids() + 2))
        # batch x sequence x acids
        return x

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self._loss(batch, pred)
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self._loss(batch, pred)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        pred_sequence = torch.argmax(pred, dim=-1) - 1
        accuracy = torch.count_nonzero(torch.eq(pred_sequence, batch)) / (
            batch.shape[0] * batch.shape[1]
        )
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)

    def _loss(self, batch, pred):
        class_first = torch.transpose(pred, 1, 2)
        return torch.mean(F.cross_entropy(class_first, batch + 1, reduction="mean"))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LinearAutoEncoder(SequenceProbabilityModel):
    def __init__(self):
        super(LinearAutoEncoder, self).__init__(
            HyperparameterDict(
                {
                    "latent": HyperparameterInt(),
                    "learning_rate": HyperparameterFloat(),
                    "dropout": HyperparameterFloat(),
                }
            ),
            _LinearAutoEncoder,
            "val_loss",
            ["loss", "val_loss", "val_accuracy"],
            patience=50,
        )

    def _get_min_epochs(self) -> int:
        return 0

    def _get_model_args(self, sequence_length: int) -> Dict[str, Any]:
        model_hyperparams = self.hyperparams["model"]
        return dict(
            sequence_length=sequence_length,
            latent=model_hyperparams["latent"].get(),
            learning_rate=model_hyperparams["learning_rate"].get(),
            dropout=model_hyperparams["dropout"].get(),
        )


class _LinearAutoEncoder(pl.LightningModule):
    def __init__(self, sequence_length: int, latent: int, learning_rate: float, dropout: float):
        super(_LinearAutoEncoder, self).__init__()
        size = sequence_length * (data.num_acids() + 2)
        self.encoder = nn.Linear(size, latent)
        self.decoder = nn.Linear(latent, size)
        self.dropout = nn.Dropout(dropout)
        self.learning_rate = learning_rate

    def encode(self, x):
        # batch x sequence
        x = F.one_hot(x + 1, num_classes=data.num_acids() + 2).to(torch.float)
        # batch x sequence x acids
        x = torch.reshape(x, (x.shape[0], -1))
        # batch x size
        z = self.encoder(x)
        # batch x latent
        return z

    def decode(self, z):
        # batch x latent
        x = self.decoder(z)
        # batch x size
        x = torch.reshape(x, (x.shape[0], -1, data.num_acids() + 2))
        # batch x sequence x acids
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

    def training_step(self, x, batch_idx):
        z = self.encode(x)
        z_dropout = self.dropout(z)
        x_hat = self.decode(z_dropout)
        loss = self._loss(x, x_hat)
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, x, batch_idx):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = self._loss(x, x_hat)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        reconstructed = torch.argmax(x_hat, dim=2) - 1
        accuracy = torch.count_nonzero(torch.eq(reconstructed, x)) / (x.shape[0] * x.shape[1])
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True)

    def _loss(self, batch, pred):
        class_first = torch.transpose(pred, 1, 2)
        return torch.mean(F.cross_entropy(class_first, batch + 1, reduction="mean"))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
