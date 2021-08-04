from typing import Any, Dict, Tuple

import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna import Trial
from torch.utils.data import DataLoader

from mutation_prediction.data import Dataset, preprocessing
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import (
    HyperparameterBool,
    HyperparameterDict,
    HyperparameterFloat,
    HyperparameterInt,
)
from mutation_prediction.models.lightning import (
    LightningModel,
    SimpleTorchDataset,
    UnsupervisedTorchDataset,
)


class MlpRegressor(LightningModel):
    def __init__(self, embedding: Embedding):
        super(MlpRegressor, self).__init__(
            HyperparameterDict(
                {
                    "layers": HyperparameterInt(),
                    "units": HyperparameterInt(),
                    "dropout": HyperparameterFloat(),
                    "validation-fraction": HyperparameterFloat(),
                    "batch-size": HyperparameterInt(),
                    "learning-rate": HyperparameterFloat(),
                    "dropout-before": HyperparameterBool(),
                }
            ),
            _MlpRegressor,
            performance_metric="rmse",
            log_metrics=["rmse", "val_rmse"],
        )
        if hasattr(embedding, "hyperparams"):
            self.hyperparams["embedding"] = embedding.hyperparams
        self.embedding = embedding

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            preprocessing.shuffle(dataset), 1 - self.hyperparams["validation-fraction"].get()
        )
        x_train = self.embedding.embed_update(train, trial=trial)
        if x_train.size == 0:
            raise optuna.exceptions.TrialPruned()
        x_val = self.embedding.embed(val)
        train_dl = DataLoader(
            SimpleTorchDataset(x_train, train.get_y()),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            SimpleTorchDataset(x_val, val.get_y()),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
        )
        model_args = dict(
            input_features=x_train.shape[1],
            units=[self.hyperparams["units"].get()] * self.hyperparams["layers"].get(),
            dropout=self.hyperparams["dropout"].get(),
            learning_rate=self.hyperparams["learning-rate"].get(),
            dropout_before=self.hyperparams["dropout-before"].get(),
        )
        return model_args, train_dl, val_dl

    def _get_min_epochs(self) -> int:
        return 100

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        x = self.embedding.embed(dataset)
        return DataLoader(
            UnsupervisedTorchDataset(x),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
        )


class _MlpRegressor(pl.LightningModule):
    def __init__(self, input_features, units, dropout, learning_rate, dropout_before=False):
        super(_MlpRegressor, self).__init__()
        layers = nn.ModuleList()
        last_size = input_features
        for u in units:
            layers.append(nn.Linear(last_size, u))
            last_size = u
        layers.append(nn.Linear(last_size, 1))
        self.layers = layers
        self.dropout = nn.Dropout(p=dropout)
        self.learning_rate = learning_rate
        self.dropout_before = dropout_before

    def forward(self, x):
        out = x
        if self.dropout_before:
            for layer in self.layers[:-1]:
                out = F.relu(layer(self.dropout(out)))
            out = self.layers[-1](self.dropout(out))
        else:
            for layer in self.layers[:-1]:
                out = self.dropout(F.relu(layer(out)))
            out = self.layers[-1](out)
        return out.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        rmse = torch.sqrt(loss)
        self.log("rmse", rmse, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        rmse = torch.sqrt(loss)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class MlpRegressor2d(LightningModel):
    def __init__(self, embedding: Embedding):
        super(MlpRegressor2d, self).__init__(
            HyperparameterDict(
                {
                    "layers": HyperparameterInt(),
                    "units": HyperparameterInt(),
                    "dropout": HyperparameterFloat(),
                    "validation-fraction": HyperparameterFloat(),
                    "batch-size": HyperparameterInt(),
                    "learning-rate": HyperparameterFloat(),
                    "dropout-before": HyperparameterBool(),
                }
            ),
            _MlpRegressor,
            performance_metric="rmse",
            log_metrics=["rmse", "val_rmse"],
        )
        if hasattr(embedding, "hyperparams"):
            self.hyperparams["embedding"] = embedding.hyperparams
        self.embedding = embedding

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            preprocessing.shuffle(dataset), 1 - self.hyperparams["validation-fraction"].get()
        )
        x_train = self.embedding.embed_update(train, trial=trial)
        if x_train.size == 0:
            raise optuna.TrialPruned()
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_val = self.embedding.embed(val)
        x_val = x_val.reshape((x_val.shape[0], -1))
        train_dl = DataLoader(
            SimpleTorchDataset(x_train, train.get_y()),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            SimpleTorchDataset(x_val, val.get_y()),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
        )
        model_args = dict(
            input_features=x_train.shape[1],
            units=[self.hyperparams["units"].get()] * self.hyperparams["layers"].get(),
            dropout=self.hyperparams["dropout"].get(),
            learning_rate=self.hyperparams["learning-rate"].get(),
            dropout_before=self.hyperparams["dropout-before"].get(),
        )
        return model_args, train_dl, val_dl

    def _get_min_epochs(self) -> int:
        return 100

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        x = self.embedding.embed(dataset)
        x = x.reshape((x.shape[0], -1))
        return DataLoader(
            UnsupervisedTorchDataset(x),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
        )
