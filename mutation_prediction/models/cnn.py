from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna import Trial
from torch.utils.data import DataLoader

import mutation_prediction.data.preprocessing as preprocessing
from mutation_prediction import data
from mutation_prediction.data import Dataset, num_acids, structure_utils
from mutation_prediction.embeddings import Embedding, aaindex
from mutation_prediction.embeddings.acids import (
    VHSE,
    AaIndex,
    AcidsOneHot,
    PcScores,
    SScales,
    ZScales,
)
from mutation_prediction.embeddings.msa import PrecomputedAutoEncoder
from mutation_prediction.embeddings.other import ConcatEmbedding
from mutation_prediction.embeddings.positional import StructurePositionalEmbeddingOld
from mutation_prediction.models import (
    HyperparameterBool,
    HyperparameterCategorical,
    HyperparameterCnnExponential,
    HyperparameterDict,
    HyperparameterFloat,
    HyperparameterInt,
    HyperparameterLayersExponential,
    HyperparameterOptional,
)
from mutation_prediction.models.lightning import (
    LightningModel,
    SimpleTorchDataset,
    UnsupervisedTorchDataset,
)


class CnnRegressorLinear(LightningModel):
    def __init__(self):
        super(CnnRegressorLinear, self).__init__(
            HyperparameterDict(
                {
                    "filters_3": HyperparameterInt(),
                    "filters_6": HyperparameterInt(),
                    "filters_9": HyperparameterInt(),
                    "filters_12": HyperparameterInt(),
                    "dropout": HyperparameterFloat(),
                    "dropout_embedding": HyperparameterFloat(),
                    "dense": HyperparameterLayersExponential(1),
                    "learning_rate": HyperparameterFloat(),
                    "embedding": HyperparameterCategorical(),
                    "train_embedding": HyperparameterBool(),
                    "validation_fraction": HyperparameterFloat(),
                    "batch_size": HyperparameterInt(),
                }
            ),
            _CnnRegressorLinear,
            performance_metric="val_rmse",
            log_metrics=["rmse", "val_rmse"],
        )

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:

        embedding_type = self.hyperparams["embedding"].get()
        if embedding_type == "one_hot":
            embedding = np.eye(num_acids())
        elif embedding_type == "zScales":
            embedding = ZScales().get_matrix()
        elif embedding_type == "VHSE":
            embedding = VHSE().get_matrix()
        elif embedding_type == "PCScores":
            embedding = PcScores().get_matrix()
        elif embedding_type == "sScales":
            embedding = SScales().get_matrix()
        elif embedding_type == "AAindex":
            embedding = AaIndex().get_matrix()
        elif embedding_type.startswith("cnnScales"):
            number = int(embedding_type[len("cnnScales") :])
            embedding = AaIndex(
                keys=[
                    "GARJ730101",
                    "PALJ810107",
                    "AURR980118",
                    "OOBM850102",
                    "GEIM800102",
                    "BUNA790103",
                    "PALJ810113",
                    "GEIM800109",
                    "ISOY800107",
                    "CHOP780215",
                    "RICJ880112",
                    "OOBM850104",
                    "COSI940101",
                    "RICJ880117",
                    "RICJ880104",
                    "WILM950104",
                    "VELV850101",
                    "ZIMJ680101",
                    "RICJ880108",
                    "NADH010107",
                    "CHOP780207",
                    "JOND920102",
                    "RACS820102",
                    "GEIM800103",
                    "AURR980101",
                    "MITS020101",
                    "SNEP660104",
                    "NAKH900107",
                    "CHAM830104",
                    "KHAG800101",
                    "RACS820103",
                    "SUEM840102",
                    "YUTK870102",
                    "RICJ880105",
                    "KUMS000101",
                    "AURR980116",
                    "ANDN920101",
                    "CRAJ730102",
                    "GEIM800106",
                    "QIAN880117",
                    "VASM830101",
                    "ROBB760110",
                    "SNEP660101",
                    "ROBB760102",
                    "AURR980102",
                    "RICJ880116",
                    "KUMS000102",
                    "KARP850103",
                    "FUKS010110",
                    "QIAN880139",
                    "CHOP780211",
                    "RICJ880114",
                    "RICJ880102",
                    "BEGF750102",
                    "RICJ880110",
                    "QIAN880101",
                    "DAYM780201",
                    "FASG760102",
                    "OOBM850103",
                    "CHOP780206",
                    "QIAN880138",
                    "AURR980103",
                    "AURR980104",
                    "BUNA790102",
                    "RICJ880101",
                    "WERD780103",
                    "FUKS010112",
                    "FUKS010109",
                    "MCMT640101",
                    "NAKH920104",
                    "TANS770108",
                    "NAGK730102",
                    "LEWP710101",
                    "CHAM830108",
                    "FINA910103",
                    "ZASB820101",
                    "CORJ870106",
                    "WEBA780101",
                    "NAKH900108",
                    "ROBB760111",
                    "CHAM830105",
                    "KARS160118",
                    "RACS820113",
                    "PALJ810111",
                    "GEOR030107",
                    "QIAN880125",
                    "JUKT750101",
                    "QIAN880104",
                    "QIAN880130",
                    "BEGF750103",
                    "KOEP990102",
                    "NAKH900101",
                    "RACS820101",
                    "CHOP780204",
                    "CEDJ970104",
                    "NAKH900111",
                    "QIAN880103",
                    "RACS820109",
                    "QIAN880102",
                    "QIAN880127",
                ][:number]
            ).get_matrix()
        else:
            raise ValueError("Unknown embedding!")
        model_args = dict(
            embedding=embedding,
            train_embedding=self.hyperparams["train_embedding"].get(),
            filters={
                3: self.hyperparams["filters_3"].get(),
                6: self.hyperparams["filters_6"].get(),
                9: self.hyperparams["filters_9"].get(),
                12: self.hyperparams["filters_12"].get(),
            },
            dense_layers=self.hyperparams["dense"].get(),
            learning_rate=self.hyperparams["learning_rate"].get(),
            dropout=self.hyperparams["dropout"].get(),
            dropout_embedding=self.hyperparams["dropout_embedding"].get(),
        )

        train, val = preprocessing.split_by_index(
            dataset, 1 - self.hyperparams["validation_fraction"].get()
        )
        dataloader_train = DataLoader(
            SimpleTorchDataset(train.get_sequences(), train.get_y(), dtype_x=torch.int),
            batch_size=self.hyperparams["batch_size"].get(),
            shuffle=True,
        )
        dataloader_val = DataLoader(
            SimpleTorchDataset(val.get_sequences(), val.get_y(), dtype_x=torch.int),
            batch_size=self.hyperparams["batch_size"].get(),
        )
        return model_args, dataloader_train, dataloader_val

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            UnsupervisedTorchDataset(dataset.get_sequences(), dtype=torch.int),
            batch_size=self.hyperparams["batch_size"].get(),
        )

    def _get_min_epochs(self) -> int:
        return 300


class _CnnRegressorLinear(pl.LightningModule):
    def __init__(
        self,
        embedding: np.ndarray,
        train_embedding: bool,
        filters: Dict[int, int],
        dense_layers: List[int],
        learning_rate: float,
        dropout: float,
        dropout_embedding: float,
    ):
        super(_CnnRegressorLinear, self).__init__()
        self.learning_rate = learning_rate
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding))
        self.embedding.weight.requires_grad = train_embedding
        self.filters = nn.ModuleList()
        self.dense = nn.ModuleList()
        total_filters = 0
        for filter_size, filter_count in filters.items():
            self.filters.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=filter_count,
                    kernel_size=(filter_size, embedding.shape[1]),
                )
            )
            total_filters += filter_count
        last_size = total_filters
        for units in dense_layers:
            self.dense.append(nn.Linear(last_size, units))
            last_size = units
        self.dropout = nn.Dropout(dropout)
        self.dropout_embedding = nn.Dropout(dropout_embedding)

    def forward(self, x):
        # batch x sequence
        x = self.embedding(x)
        # batch x sequence x features
        x = self.dropout_embedding(x)
        # batch x sequence x features
        x = x.unsqueeze(1)
        # batch x 1 x sequence x features
        filter_values = []
        for filter in self.filters:
            filtered = filter(x).squeeze(3)
            # batch x filters x sequence
            filtered = F.max_pool1d(filtered, filtered.shape[2]).squeeze(2)
            # batch x filters
            filter_values.append(F.relu(filtered))
        out = torch.cat(filter_values, dim=1)
        # batch x total_filters
        for dense in self.dense[:-1]:
            out = self.dropout(out)
            out = F.relu(dense(out))
        out = self.dropout(out)
        out = self.dense[-1](out)
        # batch x 1
        return out.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted = self(x)
        loss = F.mse_loss(predicted, y)
        self.log("rmse", torch.sqrt(loss), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        rmse = torch.sqrt(F.mse_loss(prediction, y))
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)


class CnnRegressorLinearCustomEmbedding(LightningModel):
    def __init__(self, embedding: Embedding):
        super(CnnRegressorLinearCustomEmbedding, self).__init__(
            HyperparameterDict(
                {
                    "filters_3": HyperparameterInt(),
                    "filters_6": HyperparameterInt(),
                    "filters_9": HyperparameterInt(),
                    "filters_12": HyperparameterInt(),
                    "dropout_first": HyperparameterFloat(),
                    "dropout_last": HyperparameterFloat(),
                    "dense": HyperparameterLayersExponential(1),
                    "learning_rate": HyperparameterFloat(),
                    "validation_fraction": HyperparameterFloat(),
                    "batch_size": HyperparameterInt(),
                }
            ),
            _CnnRegressorLinearCustomEmbedding,
            performance_metric="val_rmse",
            log_metrics=["rmse", "val_rmse"],
        )
        if hasattr(embedding, "hyperparams"):
            self.hyperparams["embedding"] = embedding.hyperparams
        self.embedding = embedding

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            dataset, 1 - self.hyperparams["validation_fraction"].get()
        )
        embedded_train = self.embedding.embed_update(train)
        embedded_val = self.embedding.embed(val)
        params = dict(
            input_features=embedded_train.shape[-1],  # TODO
            filters={
                3: self.hyperparams["filters_3"].get(),
                6: self.hyperparams["filters_6"].get(),
                9: self.hyperparams["filters_9"].get(),
                12: self.hyperparams["filters_12"].get(),
            },
            dense_layers=self.hyperparams["dense"].get(),
            learning_rate=self.hyperparams["learning_rate"].get(),
            dropout_first=self.hyperparams["dropout_first"].get(),
            dropout_last=self.hyperparams["dropout_last"].get(),
        )
        dataloader_train = DataLoader(
            SimpleTorchDataset(embedded_train, train.get_y(), dtype_x=torch.float32),
            batch_size=self.hyperparams["batch_size"].get(),
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        dataloader_val = DataLoader(
            SimpleTorchDataset(embedded_val, val.get_y(), dtype_x=torch.float32),
            batch_size=self.hyperparams["batch_size"].get(),
            pin_memory=True,
            num_workers=1,
        )
        return params, dataloader_train, dataloader_val

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        embedded = self.embedding.embed(dataset)
        return DataLoader(
            UnsupervisedTorchDataset(embedded, dtype=torch.float32),
            batch_size=self.hyperparams["batch_size"].get(),
        )

    def _get_min_epochs(self) -> int:
        return 300


class _CnnRegressorLinearCustomEmbedding(pl.LightningModule):
    def __init__(
        self,
        input_features: int,
        filters: Dict[int, int],
        dense_layers: List[int],
        learning_rate: float,
        dropout_first: float,
        dropout_last: float,
    ):
        super(_CnnRegressorLinearCustomEmbedding, self).__init__()
        self.learning_rate = learning_rate
        self.filters = nn.ModuleList()
        self.dense = nn.ModuleList()
        total_filters = 0
        for filter_size, filter_count in filters.items():
            self.filters.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=filter_count,
                    kernel_size=(filter_size, input_features),
                )
            )
            total_filters += filter_count
        last_size = total_filters
        for units in dense_layers:
            self.dense.append(nn.Linear(last_size, units))
            last_size = units
        self.dropout_first = nn.Dropout(dropout_first)
        self.dropout_last = nn.Dropout(dropout_last)

    def forward(self, x):
        x = self.dropout_first(x)
        # batch x sequence x features
        x = x.unsqueeze(1)
        # batch x 1 x sequence x features
        filter_values = []
        for filter in self.filters:
            filtered = filter(x).squeeze(3)
            # batch x filters x sequence
            filtered = F.max_pool1d(filtered, filtered.shape[2]).squeeze(2)
            # batch x filters
            filter_values.append(F.relu(filtered))
        out = torch.cat(filter_values, dim=1)
        # batch x total_filters
        for dense in self.dense[:-1]:
            out = self.dropout_last(out)
            out = F.relu(dense(out))
        out = self.dropout_last(out)
        out = self.dense[-1](out)
        # batch x 1
        return out.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predicted = self(x)
        loss = F.mse_loss(predicted, y)
        self.log("rmse", torch.sqrt(loss), on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        rmse = torch.sqrt(F.mse_loss(prediction, y))
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)


class CnnRegressor3d(LightningModel):
    def __init__(self):
        super(CnnRegressor3d, self).__init__(
            HyperparameterDict(
                {
                    "conv": HyperparameterCnnExponential(data.num_acids()),
                    "dense": HyperparameterLayersExponential(1),
                    "dropout": HyperparameterFloat(),
                    "validation_fraction": HyperparameterFloat(),
                    "grid_size": HyperparameterInt(),
                    "learning_rate": HyperparameterFloat(),
                    "batch_size": HyperparameterInt(),
                }
            ),
            _CnnRegressor3d,
            performance_metric="val_rmse",
            log_metrics=["rmse", "val_rmse"],
        )

    def _get_min_epochs(self) -> int:
        return 500

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        model_args = dict(
            features=self.hyperparams["conv"].get()["layers"],
            kernel_sizes=self.hyperparams["conv"].get()["kernel_sizes"],
            dense_layers=self.hyperparams["dense"].get(),
            learning_rate=self.hyperparams["learning_rate"].get(),
            dropout=self.hyperparams["dropout"].get(),
            grid_size=self.hyperparams["grid_size"].get(),
        )
        train, val = preprocessing.split_by_index(
            preprocessing.shuffle(dataset), 1 - self.hyperparams["validation_fraction"].get()
        )
        train_dataloader = DataLoader(
            _GridDataset(train, self.hyperparams["grid_size"].get()),
            shuffle=True,
            batch_size=self.hyperparams["batch_size"].get(),
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            _GridDataset(train, self.hyperparams["grid_size"].get()),
            batch_size=self.hyperparams["batch_size"].get(),
            pin_memory=True,
        )
        return model_args, train_dataloader, val_dataloader

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            _GridDataset(dataset.strip_y(), self.hyperparams["grid_size"].get()),
            batch_size=self.hyperparams["batch_size"].get(),
            pin_memory=True,
        )


class _CnnRegressor3d(pl.LightningModule):
    def __init__(
        self,
        features: List[int],
        kernel_sizes: List[int],
        dense_layers: List[int],
        learning_rate: float,
        dropout: float,
        grid_size: int,
    ):
        assert len(features) == len(kernel_sizes)
        super(_CnnRegressor3d, self).__init__()
        self.learning_rate = learning_rate
        last_features = data.num_acids()
        last_grid_size = grid_size
        self.convolutions = nn.ModuleList()
        for f, k in zip(features, kernel_sizes):
            self.convolutions.append(
                nn.Conv3d(
                    last_features, f, kernel_size=k, padding=(k - 1) // 2, padding_mode="zeros"
                )
            )
            last_features = f
            last_grid_size = last_grid_size // 2
        last_size = last_features * last_grid_size * last_grid_size * last_grid_size
        self.dense = nn.ModuleList()
        for units in dense_layers:
            self.dense.append(nn.Linear(last_size, units))
            last_size = units
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # batch x grid x grid x grid x 20
        x = torch.transpose(x, 1, 4)
        # batch x 20 x grid x grid x grid
        for convolution in self.convolutions:
            x = F.relu(convolution(x))
            x = F.max_pool3d(x, 2)
        # batch x features x subgrid x subgrid x subgrid
        x = torch.reshape(x, (x.shape[0], -1))
        # batch x size
        for dense in self.dense[:-1]:
            x = self.dropout(x)
            x = F.relu(dense(x))
        x = self.dropout(x)
        x = self.dense[-1](x)
        # batch x 1
        return torch.squeeze(x, 1)

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
        rmse = torch.sqrt(F.mse_loss(pred, y))
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class _GridDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, grid_size: int):
        self.dataset = dataset
        self.prep_values = structure_utils.prepare_gridify(dataset, grid_size)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x = torch.tensor(
            structure_utils.gridify(self.dataset, item, self.prep_values), dtype=torch.float32
        )
        if self.dataset.get_y() is None:
            return x
        else:
            return x, torch.tensor(self.dataset.get_y()[item], dtype=torch.float32)


class KCnn(LightningModel):
    def __init__(self, embedding: Embedding):
        hyperparam_dict = {
            "k": HyperparameterInt(),
            "batch-size": HyperparameterInt(),
            "validation-fraction": HyperparameterInt(),
            "filters": HyperparameterInt(),
            "dropout": HyperparameterFloat(),
            "learning-rate": HyperparameterFloat(),
            "dense": HyperparameterLayersExponential(1),
            "filter-dropout": HyperparameterFloat(),
        }
        if hasattr(embedding, "hyperparams"):
            hyperparam_dict["embedding"] = embedding.hyperparams
        super(KCnn, self).__init__(
            HyperparameterDict(hyperparam_dict),
            _KCnnRegressor,
            "val_loss",
            ["loss", "val_loss"],
        )
        self.embedding = embedding
        self.mask = None

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            preprocessing.shuffle(dataset), 1 - self.hyperparams["validation-fraction"].get()
        )
        # train_ds = _KDataSet(train, self.hyperparams["k"].get(), self.embedding.embed_update(train))
        neighborhood, distances, self.mask = structure_utils.knn(
            dataset, self.hyperparams["k"].get()
        )
        distances = distances / np.max(distances)
        train_ds = SimpleTorchDataset(
            self.embedding.embed_update(train, trial=trial), train.get_y()
        )
        example = train_ds[0]
        train_dl = DataLoader(
            train_ds,
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            # _KDataSet(val, self.hyperparams["k"].get(), self.embedding.embed(val)),
            SimpleTorchDataset(self.embedding.embed(val), val.get_y()),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
        )
        model_args = dict(
            k=self.hyperparams["k"].get(),
            n_features=example[0].shape[-1],
            filters=self.hyperparams["filters"].get(),
            dropout=self.hyperparams["dropout"].get(),
            lr=self.hyperparams["learning-rate"].get(),
            dense_units=self.hyperparams["dense"].get(),
            neighborhood=torch.tensor(neighborhood, dtype=torch.long),
            distances=torch.tensor(distances, dtype=torch.float),
            filter_dropout=self.hyperparams["filter-dropout"].get(),
        )
        return model_args, train_dl, val_dl

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            # _KDataSet(dataset, self.hyperparams["k"].get(), self.embedding.embed(dataset), y=False),
            UnsupervisedTorchDataset(self.embedding.embed(dataset)),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
        )

    def _get_min_epochs(self) -> int:
        return 0


class _KCnnRegressor(pl.LightningModule):
    def __init__(
        self,
        k: int,
        n_features: int,
        filters: int,
        dropout: float,
        lr: float,
        dense_units: List[int],
        neighborhood: torch.tensor,
        distances: torch.tensor,
        filter_dropout: float,
    ):
        super(_KCnnRegressor, self).__init__()
        self.lr = lr
        self.filters = nn.Linear(k * (n_features + 1), filters)
        dense = []
        last_size = filters
        for u in dense_units[:-1]:
            dense.append(nn.Dropout(p=dropout))
            dense.append(nn.Linear(last_size, u))
            dense.append(nn.ReLU())
            last_size = u
        dense.append(nn.Dropout(p=dropout))
        dense.append(nn.Linear(last_size, dense_units[-1]))
        self.dense = nn.Sequential(*dense)
        self.neighborhood = nn.Parameter(neighborhood, requires_grad=False)
        self.distances = nn.Parameter(distances, requires_grad=False)
        self.filter_dropout = nn.Dropout(p=filter_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch x sequence x features
        distances = self.distances.unsqueeze(0).expand(x.shape[0], -1, -1).unsqueeze(3)
        x = torch.cat([x[:, self.neighborhood, :], distances], dim=-1)
        # batch x sequence x k x features
        batch_size = x.shape[0]
        sequence_len = x.shape[1]
        x = torch.reshape(x, (batch_size * sequence_len, -1))
        # batch * sequence x k * features
        x = self.filters(x)
        # batch * sequence x filters
        x = torch.reshape(x, (batch_size, sequence_len, -1))
        # batch x sequence x filters
        x = self.filter_dropout(x)
        # batch x sequence x filters
        x = torch.max(x, dim=1)[0]
        # batch x filters
        x = F.relu(x)
        x = self.dense(x)
        # batch x 1
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.sqrt(F.mse_loss(pred, y))
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.sqrt(F.mse_loss(pred, y))
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class _KDataSet(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, k: int, embedding: np.ndarray, y: bool = True):
        self.y = dataset.get_y() if y else None
        self.k = k
        self.embedding = embedding
        self.nearest, self.distances, self.mask = structure_utils.knn(dataset, k)
        self.distances = self.distances / np.max(self.distances)

    def __len__(self) -> int:
        return len(self.embedding)

    def __getitem__(self, item) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedding = self.embedding[item][self.nearest]
        x = np.concatenate([self.distances[:, :, None], embedding], axis=-1)
        x = torch.tensor(x, dtype=torch.float)
        if self.y is not None:
            y = torch.tensor(self.y[item], dtype=torch.float)
            return x, y
        else:
            return x


class KCnnPlus(LightningModel):
    def __init__(self):
        self.embedding = ConcatEmbedding(
            AcidsOneHot(),
            PrecomputedAutoEncoder("Linear"),
            StructurePositionalEmbeddingOld(),
            configurable=True,
        )
        super(KCnnPlus, self).__init__(
            HyperparameterDict(
                {
                    "k": HyperparameterInt(),
                    "batch-size": HyperparameterInt(),
                    "validation-fraction": HyperparameterInt(),
                    "filters": HyperparameterInt(),
                    "input-dropout": HyperparameterFloat(),
                    "final-dropout": HyperparameterFloat(),
                    "learning-rate": HyperparameterFloat(),
                    "dense": HyperparameterLayersExponential(1),
                    "aaindex-reduction": HyperparameterOptional(HyperparameterInt()),
                    "embedding": self.embedding.hyperparams,
                }
            ),
            _KCnnPlus,
            "val_loss",
            ["loss", "val_loss"],
        )

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            preprocessing.shuffle(dataset), 1 - self.hyperparams["validation-fraction"].get()
        )

        train_ds = _KCnnPlusDataset(train, self.embedding.embed_update(train))
        example = train_ds[0]
        train_dl = DataLoader(
            train_ds,
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            _KCnnPlusDataset(val, self.embedding.embed(val)),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
        )

        neigborhood, distances, mask = structure_utils.knn(train, self.hyperparams["k"].get())
        distances = distances / distances.max()

        acid_embedding = aaindex.read_aaindex1()
        acid_embedding = acid_embedding - acid_embedding.mean(axis=0)
        acid_embedding = acid_embedding / np.max(np.abs(acid_embedding), axis=0)

        pair_embedding = aaindex.read_aaindex3()
        pair_embedding = pair_embedding - pair_embedding.mean(axis=(0, 1))
        pair_embedding = pair_embedding / np.max(np.abs(pair_embedding), axis=(0, 1))

        model_args = dict(
            filters=self.hyperparams["filters"].get(),
            input_dropout=self.hyperparams["input-dropout"].get(),
            final_dropout=self.hyperparams["final-dropout"].get(),
            lr=self.hyperparams["learning-rate"].get(),
            dense_units=self.hyperparams["dense"].get(),
            neighborhood=torch.tensor(neigborhood, dtype=torch.long),
            distances=torch.tensor(distances, dtype=torch.float),
            aaindex_acids=torch.tensor(acid_embedding, dtype=torch.float),
            aaindex_pairs=torch.tensor(pair_embedding, dtype=torch.float),
            aaindex_reduction=self.hyperparams["aaindex-reduction"].get(),
            sequence_features=example[0][1].shape[1],
        )

        return model_args, train_dl, val_dl

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            _KCnnPlusDataset(dataset, self.embedding.embed(dataset), y=False),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
        )

    def _get_min_epochs(self) -> int:
        return 0


class _KCnnPlus(pl.LightningModule):
    def __init__(
        self,
        filters: int,
        input_dropout: float,
        final_dropout: float,
        lr: float,
        dense_units: List[int],
        neighborhood: torch.tensor,
        distances: torch.tensor,
        aaindex_acids: torch.tensor,
        aaindex_pairs: torch.tensor,
        aaindex_reduction: int,
        sequence_features: int,
    ):
        super(_KCnnPlus, self).__init__()
        self.lr = lr
        self.neighborhood = nn.Parameter(neighborhood, requires_grad=False)
        self.distances = nn.Parameter(distances, requires_grad=False)
        self.aaindex_acids = nn.Parameter(aaindex_acids, requires_grad=False)
        self.aaindex_pairs = nn.Parameter(aaindex_pairs, requires_grad=False)
        if aaindex_reduction is not None:
            self.aaindex_reduction = nn.Linear(
                aaindex_acids.shape[1] + aaindex_pairs.shape[2], aaindex_reduction
            )
            n_features = aaindex_reduction + sequence_features + 1  # +1 for distance
        else:
            self.aaindex_reduction = None
            n_features = (
                aaindex_acids.shape[1] + aaindex_pairs.shape[2] + sequence_features + 1
            )  # +1 for distance
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.filters = nn.Linear(neighborhood.shape[1] * n_features, filters)
        dense = []
        last_size = filters
        for u in dense_units[:-1]:
            dense.append(nn.Dropout(p=final_dropout))
            dense.append(nn.Linear(last_size, u))
            dense.append(nn.ReLU())
            last_size = u
        dense.append(nn.Dropout(p=final_dropout))
        dense.append(nn.Linear(last_size, dense_units[-1]))
        self.dense = nn.Sequential(*dense)

    def forward(self, *args) -> torch.Tensor:
        if len(args) == 1:
            sequence, sequence_embedded = args[0]
        else:
            sequence, sequence_embedded = args

        # retrieve shape parameters
        batch_size = sequence.shape[0]
        sequence_len = self.neighborhood.shape[0]
        k = self.neighborhood.shape[1]

        # build full aaindex embedding
        neighbor_acids = sequence[:, self.neighborhood]
        aaindex_acids_len = self.aaindex_acids.shape[1]
        aaindex_pairs_len = self.aaindex_pairs.shape[2]
        aaindex: torch.Tensor = torch.empty(
            (batch_size, sequence_len, k, aaindex_acids_len + aaindex_pairs_len),
            dtype=sequence_embedded.dtype,
            device=self.device,
        )
        aaindex[:, :, :, :aaindex_acids_len] = self.aaindex_acids[neighbor_acids]
        aaindex[:, :, :, aaindex_acids_len:] = self.aaindex_pairs[
            neighbor_acids[:, :, 0].unsqueeze(2).expand(-1, -1, k), neighbor_acids[:, :, :]
        ]

        # reduce aaindex embedding
        if self.aaindex_reduction is not None:
            aaindex = aaindex.view(batch_size * sequence_len * k, -1)
            aaindex = self.aaindex_reduction(aaindex)
            aaindex = aaindex.view(batch_size, sequence_len, k, -1)

        # calculate sequence embedding
        sequence_embedded = sequence_embedded[:, self.neighborhood]

        # calculate distances embedding
        distances = self.distances.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(3)

        # concatenate them all together
        x = torch.cat([sequence_embedded, distances, aaindex], dim=-1)
        x = self.input_dropout(x)

        # apply filters
        x = x.view(batch_size * sequence_len, -1)
        x = self.filters(x)
        x = x.view(batch_size, sequence_len, -1)

        # max over sequence
        x = torch.max(x, dim=1)[0]

        # non-linearity
        x = F.relu(x)

        # final dense layer
        x = self.dense(x)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(*x)
        loss = torch.sqrt(F.mse_loss(pred, y))
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(*x)
        loss = torch.sqrt(F.mse_loss(pred, y))
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class _KCnnPlusDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Dataset, embedding: np.ndarray, y: bool = True):
        self.y = dataset.get_y() if y else None
        self.embedding = embedding
        self.sequences = dataset.get_sequences()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, item):
        x = (
            torch.tensor(self.sequences[item], dtype=torch.long),
            torch.tensor(self.embedding[item], dtype=torch.float),
        )
        if self.y is not None:
            y = torch.tensor(self.y[item], dtype=torch.float)
            return x, y
        else:
            return x


class KCnnMlp(LightningModel):
    def __init__(self, embedding_cnn: Embedding, embedding_mlp: Embedding):
        hyperparam_dict = {
            "k": HyperparameterInt(),
            "learning-rate": HyperparameterFloat(),
            "batch-size": HyperparameterInt(),
            "validation-fraction": HyperparameterInt(),
            "cnn-units": HyperparameterInt(),
            "cnn-layers": HyperparameterInt(),
            "cnn-dropout": HyperparameterFloat(),
            "filter-dropout": HyperparameterFloat(),
            "mlp-units": HyperparameterInt(),
            "mlp-layers": HyperparameterInt(),
            "mlp-dropout": HyperparameterFloat(),
            "final-dense": HyperparameterLayersExponential(1),
            "final-dropout": HyperparameterFloat(),
        }
        if hasattr(embedding_cnn, "hyperparams"):
            hyperparam_dict["embedding-cnn"] = embedding_cnn.hyperparams
        if hasattr(embedding_mlp, "hyperparams"):
            hyperparam_dict["embedding-mlp"] = embedding_mlp.hyperparams
        super(KCnnMlp, self).__init__(
            HyperparameterDict(hyperparam_dict),
            _KCnnMlpRegressor,
            "val_loss",
            ["loss", "val_loss"],
        )
        self.embedding_cnn = embedding_cnn
        self.embedding_mlp = embedding_mlp
        self.mask = None

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            preprocessing.shuffle(dataset), 1 - self.hyperparams["validation-fraction"].get()
        )
        neighborhood, distances, self.mask = structure_utils.knn(
            dataset,
            self.hyperparams["k"].get(),
            indices_masked=True,
        )
        distances = distances / np.max(distances)
        train_ds = _KCnnMlpDataSet(
            self.embedding_cnn.embed_update(train, trial=trial)[:, self.mask],
            self.embedding_mlp.embed_update(train, trial=trial),
            train.get_y(),
        )
        example = train_ds[0]
        train_dl = DataLoader(
            train_ds,
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            _KCnnMlpDataSet(
                self.embedding_cnn.embed(val)[:, self.mask],
                self.embedding_mlp.embed(val),
                val.get_y(),
            ),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
        )
        model_args = dict(
            k=self.hyperparams["k"].get(),
            neighborhood=torch.tensor(neighborhood, dtype=torch.long),
            distances=torch.tensor(distances, dtype=torch.float),
            lr=self.hyperparams["learning-rate"].get(),
            n_cnn_features=example[0][0].shape[-1],
            cnn_units=[self.hyperparams["cnn-units"].get()] * self.hyperparams["cnn-layers"].get(),
            cnn_dropout=self.hyperparams["cnn-dropout"].get(),
            filter_dropout=self.hyperparams["filter-dropout"].get(),
            n_mlp_features=example[0][1].shape[-1],
            mlp_units=[self.hyperparams["mlp-units"].get()] * self.hyperparams["mlp-layers"].get(),
            mlp_dropout=self.hyperparams["mlp-dropout"].get(),
            final_units=self.hyperparams["final-dense"].get(),
            final_dropout=self.hyperparams["final-dropout"].get(),
        )
        return model_args, train_dl, val_dl

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            _KCnnMlpDataSet(
                self.embedding_cnn.embed(dataset)[:, self.mask],
                self.embedding_mlp.embed(dataset),
            ),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=0,
            pin_memory=True,
        )

    def _get_min_epochs(self) -> int:
        return 0


class _KCnnMlpRegressor(pl.LightningModule):
    def __init__(
        self,
        k: int,
        neighborhood: torch.tensor,
        distances: torch.tensor,
        lr: float,
        n_cnn_features: int,
        cnn_units: List[int],
        cnn_dropout: float,
        filter_dropout: float,
        n_mlp_features: int,
        mlp_units: List[int],
        mlp_dropout: float,
        final_units: List[int],
        final_dropout: float,
    ):
        super(_KCnnMlpRegressor, self).__init__()
        self.lr = lr
        n_features = 0

        # kCNN part
        cnn = []
        last_size = n_cnn_features
        for u in cnn_units:
            cnn.append(nn.Dropout(p=cnn_dropout))
            cnn.append(_KCnnLayer(last_size, u, neighborhood, distances))
            last_size = u
            cnn.append(nn.ReLU())
        self.cnn = nn.Sequential(*cnn)
        self.filter_dropout = nn.Dropout(p=filter_dropout)
        n_features += last_size

        # MLP part
        if n_mlp_features > 0:
            mlp = []
            last_size = n_mlp_features
            for u in mlp_units:
                mlp.append(nn.Dropout(p=mlp_dropout))
                mlp.append(nn.Linear(last_size, u))
                last_size = u
                mlp.append(nn.ReLU())
            self.mlp = nn.Sequential(*mlp)
            n_features += last_size
        else:
            self.mlp = None

        # final part
        final = []
        last_size = n_features
        for u in final_units[:-1]:
            final.append(nn.Dropout(p=final_dropout))
            final.append(nn.Linear(last_size, u))
            final.append(nn.ReLU())
            last_size = u
        final.append(nn.Dropout(p=final_dropout))
        final.append(nn.Linear(last_size, final_units[-1]))
        self.final = nn.Sequential(*final)

    def forward(self, *args: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Reformat tuples if necessary
        if len(args) == 1:
            x_cnn, x_mlp = args[0]
        else:
            x_cnn, x_mlp = args

        # PART I: kCNN
        x_cnn = self.cnn(x_cnn)
        x_cnn = self.filter_dropout(x_cnn)
        x_cnn = torch.max(x_cnn, dim=1)[0]
        x_cnn = F.relu(x_cnn)

        if self.mlp is not None:
            # PART II: MLP
            x_mlp = self.mlp(x_mlp)

        # PART III: Final layers
        x = torch.cat([x_cnn, x_mlp], dim=-1)
        x = self.final(x)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.sqrt(F.mse_loss(pred, y))
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.sqrt(F.mse_loss(pred, y))
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class _KCnnMlpDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        embedding_cnn: np.ndarray,
        embedding_mlp: np.ndarray,
        y: np.ndarray = None,
    ):
        self.y = y
        self.embedding_cnn = embedding_cnn
        self.embedding_mlp = embedding_mlp

    def __len__(self) -> int:
        return len(self.embedding_cnn)

    def __getitem__(self, item):
        x = (
            torch.tensor(self.embedding_cnn[item], dtype=torch.float),
            torch.tensor(self.embedding_mlp[item], dtype=torch.float),
        )
        if self.y is not None:
            y = torch.tensor(self.y[item], dtype=torch.float)
            return x, y
        else:
            return x


class _KCnnLayer(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        neighborhood: torch.Tensor,
        distances: torch.Tensor = None,
    ):
        super(_KCnnLayer, self).__init__()
        self.neighborhood = nn.Parameter(neighborhood, requires_grad=False)
        self.distances = (
            nn.Parameter(distances, requires_grad=False) if distances is not None else None
        )
        k = self.neighborhood.shape[1]
        if distances is not None:
            input_features += 1
        self.filters = nn.Linear(k * input_features, output_features)

    def forward(self, x: torch.Tensor):
        # batch x sequence x input_features
        if self.distances is not None:
            distances = self.distances.unsqueeze(0).expand(x.shape[0], -1, -1).unsqueeze(3)
            x = torch.cat([x[:, self.neighborhood, :], distances], dim=-1)
        batch_size = x.shape[0]
        sequence_len = x.shape[1]
        # batch x sequence x k x input_features
        x = torch.reshape(x, (batch_size * sequence_len, -1))
        # batch * sequence x k * input_features
        x = self.filters(x)
        # batch * sequence x output_features
        x = torch.reshape(x, (batch_size, sequence_len, -1))
        # batch x sequence x output_features
        return x
