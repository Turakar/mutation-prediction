import functools
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from optuna import Trial
from torch.utils.data import DataLoader as TorchDataLoader
from torch_cluster import knn_graph, radius_graph
from torch_geometric.data import Data, DataLoader

import mutation_prediction.data.structure_utils as structure_utils
from mutation_prediction.data import Dataset, preprocessing
from mutation_prediction.embeddings import aaindex
from mutation_prediction.models import (
    Hyperparameter,
    HyperparameterBool,
    HyperparameterCategorical,
    HyperparameterDict,
    HyperparameterFloat,
    HyperparameterInt,
)
from mutation_prediction.models.lightning import LightningModel


def make_embeddings():
    acid_embedding = aaindex.read_aaindex1()
    acid_embedding = acid_embedding - acid_embedding.mean(axis=0)
    acid_embedding = acid_embedding / np.max(np.abs(acid_embedding), axis=0)

    pair_embedding = aaindex.read_aaindex3()
    pair_embedding = pair_embedding - pair_embedding.mean(axis=(0, 1))
    pair_embedding = pair_embedding / np.max(np.abs(pair_embedding), axis=(0, 1))

    return acid_embedding, pair_embedding


def calc_closeness(dataset: Dataset, edges: np.ndarray, positions: np.ndarray) -> np.ndarray:
    sequence_length = len(dataset.get_sequence())
    distances = np.zeros((sequence_length, sequence_length), dtype=np.float32)
    for i in range(sequence_length):
        for j in range(sequence_length):
            distances[i, j] = np.linalg.norm(positions[i] - positions[j])
    closeness = (1 / (distances + 1))[edges[0], edges[1]]
    closeness = closeness / closeness.max()
    return closeness


class HyperparameterGraphMethod(Hyperparameter):
    def __init__(self):
        super(HyperparameterGraphMethod, self).__init__()

    def _suggest(self, trial: Trial, params: Dict[str, Any]) -> Any:
        method = trial.suggest_categorical(self.id + "_method", ["knn", "radius"])
        if method == "knn":
            k = trial.suggest_int(self.id + "_k", **params["k"])
            return method, k
        elif method == "radius":
            radius = trial.suggest_float(self.id + "_radius", **params["radius"])
            return method, radius
        else:
            raise KeyError("Unknown method!")

    def to_function(self) -> Callable:
        method, param = self.get()
        if method == "knn":
            return functools.partial(knn_graph, k=param)
        elif method == "radius":
            return functools.partial(radius_graph, r=param)
        else:
            raise KeyError("Unknown method!")


class GraphTransformerRegressor(LightningModel):
    def __init__(self):
        super(GraphTransformerRegressor, self).__init__(
            HyperparameterDict(
                {
                    "graph": HyperparameterGraphMethod(),
                    "batch-size": HyperparameterInt(),
                    "validation-fraction": HyperparameterFloat(),
                    "hidden": HyperparameterInt(),
                    "heads": HyperparameterInt(),
                    "input-dropout": HyperparameterFloat(),
                    "graph-dropout": HyperparameterFloat(),
                    "final-dropout": HyperparameterFloat(),
                    "layers": HyperparameterInt(),
                    "learning-rate": HyperparameterFloat(),
                }
            ),
            _GraphTransformerRegressor,
            "loss",
            ["loss", "val_loss"],
            patience=300,
        )

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            dataset, 1 - self.hyperparams["validation-fraction"].get()
        )
        train_ds = self._make_dataset(train)
        train_dl = DataLoader(
            train_ds,
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=1,
            pin_memory=True,
            shuffle=True,
        )
        val_ds = self._make_dataset(val)
        val_dl = DataLoader(
            val_ds,
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=1,
            pin_memory=True,
        )
        example = train_ds[0]
        model_args = dict(
            node_features=example.num_node_features,
            edge_features=example.num_edge_features,
            hidden=self.hyperparams["hidden"].get(),
            heads=self.hyperparams["heads"].get(),
            input_dropout=self.hyperparams["input-dropout"].get(),
            graph_dropout=self.hyperparams["graph-dropout"].get(),
            final_dropout=self.hyperparams["final-dropout"].get(),
            layers=self.hyperparams["layers"].get(),
            lr=self.hyperparams["learning-rate"].get(),
        )
        return model_args, train_dl, val_dl

    def _make_predict_data_loader(self, dataset: Dataset) -> TorchDataLoader:
        return DataLoader(
            self._make_dataset(dataset),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=1,
            pin_memory=True,
        )

    def _get_min_epochs(self) -> int:
        return 0

    def _make_dataset(self, dataset: Dataset) -> List[Data]:
        positions = structure_utils.get_positions(dataset)
        pos = torch.tensor(positions, dtype=torch.float)
        edges = self.hyperparams["graph"].to_function()(pos)
        closeness = calc_closeness(dataset, edges, positions)
        acid_embedding, pair_embedding = make_embeddings()

        graphs = []
        for i, sequence in enumerate(dataset.get_sequences()):
            x = torch.tensor(acid_embedding[sequence], dtype=torch.float)
            y = torch.tensor(dataset.get_y()[i], dtype=torch.float)
            pair_attr = pair_embedding[sequence[edges[0]], sequence[edges[1]]]
            edge_attr = torch.tensor(
                np.concatenate([pair_attr, closeness[..., None]], axis=-1), dtype=torch.float
            )
            data = Data(x=x, pos=pos, edge_index=edges, y=y, edge_attr=edge_attr)
            graphs.append(data)
        return graphs


class _GraphTransformerRegressor(pl.LightningModule):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden: int,
        heads: int,
        input_dropout: float,
        graph_dropout: float,
        final_dropout: float,
        layers: int,
        lr: float,
    ):
        super(_GraphTransformerRegressor, self).__init__()
        last_size = node_features
        self.conv = nn.ModuleList()
        for i in range(layers):
            self.conv.append(
                gnn.TransformerConv(
                    last_size, hidden, heads=heads, edge_dim=edge_features, dropout=graph_dropout
                )
            )
            last_size = hidden * heads
        self.dense = nn.Linear(last_size, 1)
        self.input_dropout = nn.Dropout(p=input_dropout)
        self.final_dropout = nn.Dropout(p=final_dropout)
        self.lr = lr

    def forward(self, batch):
        x = batch.x
        x = self.input_dropout(x)
        for conv in self.conv:
            x = conv(x, batch.edge_index, batch.edge_attr)
        x = gnn.global_max_pool(x, batch.batch)
        x = self.final_dropout(x)
        x = self.dense(x)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class GraphTransformerSimpleRegressor(LightningModel):
    def __init__(self):
        super(GraphTransformerSimpleRegressor, self).__init__(
            HyperparameterDict(
                {
                    "graph": HyperparameterGraphMethod(),
                    "batch-size": HyperparameterInt(),
                    "validation-fraction": HyperparameterFloat(),
                    "acid-reduction": HyperparameterInt(),
                    "pair-reduction": HyperparameterInt(),
                    "capacity": HyperparameterInt(),
                    "heads": HyperparameterInt(),
                    "graph-dropout": HyperparameterFloat(),
                    "final-dropout": HyperparameterFloat(),
                    "learning-rate": HyperparameterFloat(),
                }
            ),
            _GraphTransformerSimpleRegressor,
            "loss",
            ["loss", "val_loss"],
            patience=100,
        )

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            dataset, 1 - self.hyperparams["validation-fraction"].get()
        )
        train_dl = DataLoader(
            self._make_dataset(train),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=1,
            pin_memory=True,
            shuffle=True,
        )
        val_dl = DataLoader(
            self._make_dataset(val),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=1,
            pin_memory=True,
        )
        acid_embedding, pair_embedding = make_embeddings()
        model_args = dict(
            acid_embedding=torch.tensor(acid_embedding, dtype=torch.float32),
            pair_embedding=torch.tensor(pair_embedding, dtype=torch.float32),
            acid_reduction=self.hyperparams["acid-reduction"].get(),
            pair_reduction=self.hyperparams["pair-reduction"].get(),
            edge_dim=1,
            hidden=self.hyperparams["capacity"].get() // self.hyperparams["heads"].get(),
            heads=self.hyperparams["heads"].get(),
            graph_dropout=self.hyperparams["graph-dropout"].get(),
            final_dropout=self.hyperparams["final-dropout"].get(),
            lr=self.hyperparams["learning-rate"].get(),
        )
        return model_args, train_dl, val_dl

    def _make_predict_data_loader(self, dataset: Dataset) -> TorchDataLoader:
        return DataLoader(
            self._make_dataset(dataset),
            batch_size=self.hyperparams["batch-size"].get(),
            num_workers=1,
            pin_memory=True,
        )

    def _get_min_epochs(self) -> int:
        return 0

    def _make_dataset(self, dataset: Dataset) -> List[Data]:
        positions = structure_utils.get_positions(dataset)
        pos = torch.tensor(positions, dtype=torch.float)
        edges = self.hyperparams["graph"].to_function()(pos)
        closeness = calc_closeness(dataset, edges, positions)

        graphs = []
        for i, sequence in enumerate(dataset.get_sequences()):
            x = torch.tensor(sequence, dtype=torch.long)
            y = torch.tensor(dataset.get_y()[i], dtype=torch.float)
            edge_attr = torch.tensor(closeness[..., None], dtype=torch.float)
            data = Data(x=x, pos=pos, edge_index=edges, y=y, edge_attr=edge_attr)
            graphs.append(data)
        return graphs


class _GraphTransformerSimpleRegressor(pl.LightningModule):
    def __init__(
        self,
        acid_embedding: torch.tensor,
        pair_embedding: torch.tensor,
        acid_reduction: int,
        pair_reduction: int,
        edge_dim: int,
        hidden: int,
        heads: int,
        graph_dropout: float,
        final_dropout: float,
        lr: float,
    ):
        super(_GraphTransformerSimpleRegressor, self).__init__()
        self.edge_dim = edge_dim  # acid distance
        self.lr = lr

        # input reduction
        self.acid_embedding = nn.Parameter(acid_embedding, requires_grad=False)
        self.pair_embedding = nn.Parameter(pair_embedding, requires_grad=False)
        self.acid_reduction = nn.Linear(acid_embedding.shape[1], acid_reduction)
        self.pair_reduction = nn.Linear(pair_embedding.shape[2], pair_reduction)

        # graph convolution
        self.conv = gnn.TransformerConv(
            acid_reduction,
            hidden,
            heads=heads,
            edge_dim=self.edge_dim + pair_reduction,
            dropout=graph_dropout,
        )

        # final dense layer
        self.final_dropout = nn.Dropout(p=final_dropout)
        self.dense = nn.Linear(hidden * heads, 1)

    def forward(self, batch):
        # input reduction
        acids = self.acid_reduction(self.acid_embedding)
        pairs = torch.reshape(self.pair_embedding, (-1, self.pair_embedding.shape[-1]))
        pairs = self.pair_reduction(pairs)
        pairs = torch.reshape(
            pairs, (self.pair_embedding.shape[0], self.pair_embedding.shape[1], -1)
        )
        sequence = batch.x
        x = acids[sequence]
        e = torch.cat(
            [
                batch.edge_attr,
                pairs[sequence[batch.edge_index[0]], sequence[batch.edge_index[1]]],
            ],
            dim=-1,
        )

        # graph convolution
        x = self.conv(x, batch.edge_index, e)
        x = gnn.global_max_pool(x, batch.batch)

        # final dense layer
        x = self.final_dropout(x)
        x = self.dense(x)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class GraphConvRegressor(LightningModel):
    def __init__(self):
        super(GraphConvRegressor, self).__init__(
            HyperparameterDict(
                {
                    "graph": HyperparameterGraphMethod(),
                    "batch-size": HyperparameterInt(),
                    "hidden": HyperparameterInt(),
                    "aggregation": HyperparameterCategorical(),
                    "conv-dropout": HyperparameterFloat(),
                    "final-dropout": HyperparameterFloat(),
                    "learning-rate": HyperparameterFloat(),
                    "weigh": HyperparameterBool(),
                    "rewire": HyperparameterBool(),
                    "validation-fraction": HyperparameterFloat(),
                    "layers": HyperparameterInt(),
                }
            ),
            _GraphConvRegressor,
            performance_metric="loss",
            log_metrics=["loss", "val_loss"],
        )

    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        train, val = preprocessing.split_by_index(
            preprocessing.shuffle(dataset),
            training_fraction=1 - self.hyperparams["validation-fraction"].get(),
        )
        train_ds = self._make_dataset(train)
        example = train_ds[0]
        train_dl = DataLoader(
            train_ds,
            batch_size=self.hyperparams["batch-size"].get(),
            shuffle=True,
            pin_memory=True,
            num_workers=1,
        )
        val_dl = DataLoader(
            self._make_dataset(val),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
            num_workers=1,
        )
        model_args = dict(
            node_features=example.num_node_features,
            hidden=self.hyperparams["hidden"].get(),
            layers=self.hyperparams["layers"].get(),
            graph_fn=self.hyperparams["graph"].to_function(),
            aggregation=self.hyperparams["aggregation"].get(),
            conv_dropout=self.hyperparams["conv-dropout"].get(),
            final_dropout=self.hyperparams["final-dropout"].get(),
            lr=self.hyperparams["learning-rate"].get(),
            weigh=self.hyperparams["weigh"].get(),
            rewire=self.hyperparams["rewire"].get(),
        )
        return model_args, train_dl, val_dl

    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            self._make_dataset(dataset),
            batch_size=self.hyperparams["batch-size"].get(),
            pin_memory=True,
            num_workers=1,
        )

    def _get_min_epochs(self) -> int:
        return 0

    def _make_dataset(self, dataset: Dataset) -> List[Data]:
        positions = structure_utils.get_positions(dataset)
        pos = torch.tensor(positions, dtype=torch.float)
        edges = self.hyperparams["graph"].to_function()(pos)
        acid_embedding, _ = make_embeddings()
        closeness = calc_closeness(dataset, edges, positions)

        graphs = []
        for i, sequence in enumerate(dataset.get_sequences()):
            x = torch.tensor(acid_embedding[sequence], dtype=torch.float)
            y = torch.tensor(dataset.get_y()[i], dtype=torch.float)
            edge_weight = torch.tensor(closeness, dtype=torch.float)
            data = Data(x=x, pos=pos, edge_index=edges, y=y, edge_weight=edge_weight)
            graphs.append(data)
        return graphs


class _GraphConvRegressor(pl.LightningModule):
    def __init__(
        self,
        node_features: int,
        hidden: int,
        layers: int,
        graph_fn: Callable,
        aggregation: str,
        conv_dropout: float,
        final_dropout: float,
        lr: float,
        weigh: bool,
        rewire: bool,
    ):
        super(_GraphConvRegressor, self).__init__()
        self.graph_fn = graph_fn
        self.lr = lr
        self.weigh = weigh
        self.rewire = rewire

        # graph convolution
        self.conv_dropout = nn.Dropout(p=conv_dropout)
        self.conv = nn.ModuleList()
        last_size = node_features
        for _ in range(layers):
            self.conv.append(
                gnn.GraphConv(
                    last_size,
                    hidden,
                    aggr=aggregation,
                )
            )
            last_size = hidden

        # final dense layer
        self.final_dropout = nn.Dropout(p=final_dropout)
        self.dense = nn.Linear(last_size, 1)

    def forward(self, batch):
        # graph convolution
        x = batch.x
        e = batch.edge_index
        w = batch.edge_weight if self.weigh else None
        x = self.conv_dropout(x)
        x = self.conv[0](x, e, w)
        for conv in self.conv[1:]:
            if self.rewire:
                e = self.graph_fn(x, batch=batch.batch)
                if self.weigh:
                    w = 1 / (torch.linalg.norm(x[e[0]] - x[e[1]]) + 1)
            x = self.conv_dropout(x)
            x = conv(x, e, w)
        x = gnn.global_max_pool(x, batch.batch)

        # final dense layer
        x = self.final_dropout(x)
        x = self.dense(x)
        return x.squeeze(1)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.mse_loss(pred, batch.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
