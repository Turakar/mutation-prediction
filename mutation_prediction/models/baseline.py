from typing import List, Tuple, Union

import glmnet
import numpy as np
import optuna.exceptions
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost
from sklearn import ensemble, svm

import mutation_prediction.node as node
from mutation_prediction import utils
from mutation_prediction.embeddings import Embedding
from mutation_prediction.models import (
    FixedEmbeddingModel,
    HyperparameterCategorical,
    HyperparameterDict,
    HyperparameterFloat,
    HyperparameterInt,
)


class SvmRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(SvmRegressor, self).__init__(
            HyperparameterDict({"sigma": HyperparameterFloat(), "C": HyperparameterFloat()}),
            embedding,
        )
        self.model = None

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        self.model = svm.SVR(gamma=self.hyperparams["sigma"].get(), C=self.hyperparams["C"].get())
        self.model.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class RandomForestRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(RandomForestRegressor, self).__init__(
            HyperparameterDict(
                {"n_estimators": HyperparameterInt(), "max_features": HyperparameterCategorical()}
            ),
            embedding,
        )
        self.model = None

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        self.model = ensemble.RandomForestRegressor(
            n_estimators=self.hyperparams["n_estimators"].get(),
            max_features=self.hyperparams["max_features"].get(),
        )
        self.model.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class GlmnetRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(GlmnetRegressor, self).__init__(
            HyperparameterDict({"alpha": HyperparameterFloat(), "lambda": HyperparameterFloat()}),
            embedding,
        )

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        if x.size == 0:
            raise optuna.exceptions.TrialPruned()
        self.model = glmnet.linear.ElasticNet(
            lambda_path=[self.hyperparams["lambda"].get(), 0],
            n_splits=0,
            alpha=self.hyperparams["alpha"].get(),
        )
        self.model.fit(x, y)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x, lamb=self.hyperparams["lambda"].get())


class XGBoostRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding):
        super(XGBoostRegressor, self).__init__(
            HyperparameterDict(
                {
                    "nrounds": HyperparameterInt(),
                    "max_depth": HyperparameterInt(),
                    "eta": HyperparameterFloat(),
                    "colsample_bytree": HyperparameterFloat(),
                    "gamma": HyperparameterFloat(),
                    "subsample": HyperparameterFloat(),
                    "min_child_weight": HyperparameterInt(),
                }
            ),
            embedding,
        )
        self.model: Union[xgboost.Booster, None] = None

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        x = xgboost.DMatrix(x, label=y)
        self.model = xgboost.train(
            {
                "max_depth": self.hyperparams["max_depth"].get(),
                "eta": self.hyperparams["eta"].get(),
                "colsample_bytree": self.hyperparams["colsample_bytree"].get(),
                "gamma": self.hyperparams["gamma"].get(),
                "subsample": self.hyperparams["subsample"].get(),
                "min_child_weight": self.hyperparams["min_child_weight"].get(),
            },
            x,
            self.hyperparams["nrounds"].get(),
        )

    def _predict(self, x: np.ndarray) -> np.ndarray:
        x = xgboost.DMatrix(x)
        y = self.model.predict(x)
        return y


class _TorchMlpRegressor(pl.LightningModule):
    def __init__(self, units, input_size, init_scale, learning_rate, momentum):
        super(_TorchMlpRegressor, self).__init__()

        def make_layer(last_size, u):
            layer = nn.Linear(last_size, u)
            nn.init.uniform_(layer.weight, a=-init_scale, b=init_scale)
            return layer

        self.layers = nn.ModuleList()
        last_size = input_size
        for u in units:
            self.layers.append(make_layer(last_size, u))
            last_size = u
        self.layers.append(make_layer(last_size, 1))

        self.learning_rate = learning_rate
        self.momentum = momentum

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)
            out = F.relu(out)
        out = self.layers[-1](out)
        return out.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted = self(x)
        loss = F.mse_loss(predicted, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        self.log("val_rmse", torch.sqrt(loss))


class _TorchDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item], dtype=torch.float)
        y = torch.tensor(self.y[item], dtype=torch.float)
        return x, y


class MlpRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding, max_epochs=5000, validation_fraction=0.4):
        super(MlpRegressor, self).__init__(
            HyperparameterDict(
                {
                    "model": HyperparameterCategorical(),
                    "init_scale": HyperparameterFloat(),
                    "batch_size": HyperparameterInt(),
                    "learning_rate": HyperparameterFloat(),
                    "momentum": HyperparameterFloat(),
                    "optimizer": HyperparameterCategorical(),
                    "eval_metric": HyperparameterCategorical(),
                    "epoch": HyperparameterInt(),
                }
            ),
            embedding,
            normalize_x=True,
            normalize_y=True,
        )
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        model_id = self.hyperparams["model"].get()
        if model_id == "model_11":
            units = [128]
        elif model_id == "model_12":
            units = [256]
        elif model_id == "model_21":
            units = [128, 64]
        elif model_id == "model_22":
            units = [256, 128]
        else:
            raise NotImplementedError("Unknown model!")
        assert self.hyperparams["eval_metric"].get() == "mse"
        assert self.hyperparams["optimizer"].get() == "SGD"
        self.model = _TorchMlpRegressor(
            units,
            x.shape[1],
            self.hyperparams["init_scale"].get(),
            self.hyperparams["learning_rate"].get(),
            self.hyperparams["momentum"].get(),
        )

        # pivot = int(x.shape[0] * (1 - self.validation_fraction))
        # dataloader_train = torch.utils.data.DataLoader(
        #     _TorchDataset(x[:pivot], y[:pivot]),
        #     batch_size=self.hyperparams["batch_size"].get(),
        #     shuffle=True,
        # )
        # dataloader_validation = torch.utils.data.DataLoader(
        #     _TorchDataset(x[pivot:], y[pivot:]),
        #     batch_size=64,
        # )
        #
        # early_stopping = pl.callbacks.EarlyStopping(monitor="val_rmse", patience=50)
        # trainer = pl.Trainer(
        #     callbacks=[early_stopping],
        #     checkpoint_callback=False,
        #     logger=False,
        #     max_epochs=self.max_epochs,
        #     gpus=self.gpus,
        #     weights_summary=None,
        #     progress_bar_refresh_rate=0,
        # )
        # trainer.fit(self.model, dataloader_train, dataloader_validation)

        dataloader = torch.utils.data.DataLoader(
            _TorchDataset(x, y), batch_size=self.hyperparams["batch_size"].get(), shuffle=True
        )
        trainer = pl.Trainer(
            callbacks=[],
            checkpoint_callback=False,
            logger=False,
            max_epochs=self.hyperparams["epoch"].get(),
            gpus=node.get_pl_gpus(),
            weights_summary=None,
            progress_bar_refresh_rate=0,
        )
        trainer.fit(self.model, dataloader)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            self.model.eval()
            x_tensor = torch.tensor(x, dtype=torch.float, requires_grad=False)
            prediction = self.model(x_tensor).numpy()
            return prediction


class _TorchCnnRegressor(pl.LightningModule):
    def __init__(
        self,
        input_length: int,
        input_features: int,
        conv_layers: List[Tuple[int, int, int, int]],
        dense_layers: List[int],
        learning_rate: float,
        momentum: float,
        initialization,
    ):
        super(_TorchCnnRegressor, self).__init__()
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.dense = nn.ModuleList()
        last_length = input_length
        last_features = input_features
        for filters, kernel_size, stride, pool_size in conv_layers:
            conv = nn.Conv1d(last_features, filters, kernel_size=kernel_size, stride=stride)
            self.convs.append(conv)
            initialization(conv.weight)
            last_length = utils.conv1d_output_length(last_length, kernel_size, stride, 0, 1)
            last_features = filters
            self.pools.append(nn.MaxPool1d(pool_size))
            last_length = utils.conv1d_output_length(last_length, pool_size, pool_size, 0, 1)
        last_size = last_length * last_features
        for units in dense_layers:
            dense = nn.Linear(last_size, units)
            self.dense.append(dense)
            initialization(dense.weight)
            last_size = units
        self.dense.append(nn.Linear(last_size, 1))
        self.learning_rate = learning_rate
        self.momentum = momentum

    def forward(self, x):
        # batch x sequence x features
        x = torch.transpose(x, 1, 2)
        # batch x features x sequence
        for conv, pool in zip(self.convs, self.pools):
            x = pool(F.relu(conv(x)))
        # batch x features x last_length
        x = torch.reshape(x, (x.shape[0], -1))
        # batch x last_size
        for dense in self.dense[:-1]:
            x = F.relu(dense(x))
        x = self.dense[-1](x)
        # batch x 1
        return torch.squeeze(x, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted = self(x)
        loss = F.mse_loss(predicted, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = F.mse_loss(prediction, y)
        self.log("val_rmse", torch.sqrt(loss))


class CnnRegressor(FixedEmbeddingModel):
    def __init__(self, embedding: Embedding, max_epochs=5000, validation_fraction=0.4):
        super(CnnRegressor, self).__init__(
            HyperparameterDict(
                {
                    "batch_size": HyperparameterInt(),
                    "learning_rate": HyperparameterFloat(),
                    "model": HyperparameterCategorical(),
                    "epoch": HyperparameterInt(),
                    "optimizer": HyperparameterCategorical(),
                    "momentum": HyperparameterFloat(),
                    "objective": HyperparameterCategorical(),
                    "initialization": HyperparameterCategorical(),
                }
            ),
            embedding,
            normalize_x=True,
            normalize_y=True,
            linearize_x=False,
        )
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction

    def _fit(self, x: np.ndarray, y: np.ndarray, trial=None):
        model_id = self.hyperparams["model"].get()
        assert self.hyperparams["initialization"].get() == "HeNormal"
        initialization = nn.init.kaiming_normal_
        if model_id == "cnn_1":
            conv_layers = [(64, 4, 1, 2), (32, 6, 3, 2)]
            dense_layers = [64, 32]
        elif model_id == "cnn_3":
            conv_layers = [(256, 10, 2, 2), (128, 8, 2, 2), (128, 6, 3, 2)]
            dense_layers = [128, 64]
        elif model_id == "cnn_5":
            conv_layers = [(256, 10, 5, 2), (128, 6, 3, 2)]
            dense_layers = [256]
        elif model_id == "cnn_6":
            conv_layers = [(256, 10, 5, 2)]
            dense_layers = [128, 64]
        elif model_id == "cnn_8":
            conv_layers = [(128, 4, 1, 2), (128, 4, 1, 2), (128, 10, 2, 2), (64, 10, 5, 2)]
            dense_layers = [128, 64]
        elif model_id == "cnn_9":
            conv_layers = [(256, 10, 5, 2), (128, 6, 3, 2)]
            dense_layers = [128, 64]
            initialization = nn.init.kaiming_uniform_
        else:
            raise NotImplementedError("Unknown model!")
        assert self.hyperparams["objective"].get() == "mse"
        assert self.hyperparams["optimizer"].get() == "nesterov_momentum"
        self.model = _TorchCnnRegressor(
            x.shape[1],
            x.shape[2],
            conv_layers,
            dense_layers,
            self.hyperparams["learning_rate"].get(),
            self.hyperparams["momentum"].get(),
            initialization,
        )

        # pivot = int(x.shape[0] * (1 - self.validation_fraction))
        # dataloader_train = torch.utils.data.DataLoader(
        #     _TorchDataset(x[:pivot], y[:pivot]),
        #     batch_size=self.hyperparams["batch_size"].get(),
        #     shuffle=True,
        # )
        # dataloader_validation = torch.utils.data.DataLoader(
        #     _TorchDataset(x[pivot:], y[pivot:]),
        #     batch_size=64,
        # )
        #
        # early_stopping = pl.callbacks.EarlyStopping(monitor="val_rmse", patience=50)
        # trainer = pl.Trainer(
        #     callbacks=[early_stopping],
        #     checkpoint_callback=False,
        #     logger=False,
        #     max_epochs=self.max_epochs,
        #     gpus=self.gpus,
        #     weights_summary=None,
        #     progress_bar_refresh_rate=0,
        # )
        # trainer.fit(self.model, dataloader_train, dataloader_validation)
        dataloader = torch.utils.data.DataLoader(
            _TorchDataset(x, y), batch_size=self.hyperparams["batch_size"].get(), shuffle=True
        )
        trainer = pl.Trainer(
            callbacks=[],
            checkpoint_callback=False,
            logger=False,
            max_epochs=self.hyperparams["epoch"].get(),
            gpus=node.get_pl_gpus(),
            weights_summary=None,
            progress_bar_refresh_rate=0,
        )
        trainer.fit(self.model, dataloader)

    def _predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            self.model.eval()
            x_tensor = torch.tensor(x, dtype=torch.float, requires_grad=False)
            prediction = self.model(x_tensor).numpy()
            return prediction
