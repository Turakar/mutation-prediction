import abc
import gc
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch
from optuna import Trial
from torch.utils import data as torch_data
from torch.utils.data import DataLoader

from mutation_prediction import node
from mutation_prediction.data import Dataset
from mutation_prediction.models import HyperparameterDict, Model


class LightningModel(Model):
    def __init__(
        self,
        hyperparams: HyperparameterDict,
        model_cls: Type[pl.LightningModule],
        performance_metric: str,
        log_metrics: List[str],
        patience: int = 100,
        max_epochs: int = 10000,
    ):
        super(LightningModel, self).__init__(hyperparams)
        self.model_cls = model_cls
        self.performance_metric = performance_metric
        self.log_metrics = log_metrics
        self.patience = patience
        self.max_epochs = max_epochs
        self.model: Optional[pl.LightningModule] = None
        self.model_params: Optional[Dict[str, Any]] = None

    def fit(self, dataset: Dataset, trial: Trial = None, verbose: bool = False):
        model_params, train, val = self._prepare_train(dataset)
        self.model_params = model_params
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
        if not verbose:
            trainer_args = dict(
                logger=False,
            )
        else:
            trainer_args = dict()
        with tempfile.TemporaryDirectory() as temp:
            trainer = pl.Trainer(
                callbacks=callbacks,
                max_epochs=self.max_epochs,
                gpus=node.get_pl_gpus(),
                default_root_dir=temp,
                progress_bar_refresh_rate=0,
                **trainer_args,
            )
            gc.collect()
            trainer.fit(self.model, train, val)
            self.model = self.model.load_from_checkpoint(
                checkpoint_callback.best_model_path, **model_params
            )

    def predict(self, dataset: Dataset) -> np.ndarray:
        with torch.no_grad():
            predictions = np.zeros(len(dataset), dtype=np.float32)
            self.model.eval()
            index = 0
            for i, batch in enumerate(self._make_predict_data_loader(dataset)):
                prediction = self.model(batch).cpu().numpy()
                length = len(prediction)
                predictions[index : index + length] = prediction
                index += length
            assert index == len(predictions)
            return predictions

    @abc.abstractmethod
    def _prepare_train(
        self, dataset: Dataset, trial: Trial = None
    ) -> Tuple[Dict[str, Any], DataLoader, DataLoader]:
        pass

    @abc.abstractmethod
    def _make_predict_data_loader(self, dataset: Dataset) -> DataLoader:
        pass

    @abc.abstractmethod
    def _get_min_epochs(self) -> int:
        pass

    def save(self, path: str):
        data = dict(
            hyperparams=self.hyperparams.get(),
            model_params=self.model_params,
            state_dict=self.model.state_dict(),
        )
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path)
        self.hyperparams.set(data["hyperparams"])
        self.model_params = data["model_params"]
        self.model = self.model_cls(**self.model_params)
        self.model.load_state_dict(data["state_dict"])
        self.model.eval()


class SimpleTorchDataset(torch_data.Dataset):
    def __init__(self, x, y, dtype_x=torch.float32, dtype_y=torch.float32):
        self.x = x
        self.y = y
        self.dtype_x = dtype_x
        self.dtype_y = dtype_y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item], dtype=self.dtype_x)
        y = torch.tensor(self.y[item], dtype=self.dtype_y)
        return x, y


class UnsupervisedTorchDataset(torch_data.Dataset):
    def __init__(self, x, dtype=torch.float32):
        self.x = x
        self.dtype = dtype

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = torch.tensor(self.x[item], dtype=self.dtype)
        return x


class TrialUserAttrsCallback(pl.callbacks.Callback):
    def __init__(self, trial: Trial, attrs: List[str], prefix: str = "", every_n_epochs: int = 1):
        self.trial = trial
        self.attrs = attrs
        self.prefix = prefix
        self.every_n_epochs = every_n_epochs
        self.logs = {}

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = int(trainer.logged_metrics.get("epoch").item())
        for attr in self.attrs:
            user_attr = "log_" + self.prefix + attr
            value = trainer.logged_metrics.get(attr).item()
            log: List[List[Any]]
            if user_attr not in self.logs:
                if user_attr not in self.trial.user_attrs:
                    self.logs[user_attr] = []
                else:
                    self.logs[user_attr] = self.trial.user_attrs[user_attr]
            log = self.logs[user_attr]
            if epoch == 0:
                log.append([])
            log[-1].append(value)
            if epoch % self.every_n_epochs == 0:
                self.trial.set_user_attr(user_attr, log)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for key, log in self.logs.items():
            self.trial.set_user_attr(key, log)


# heavily based on
# https://github.com/PyTorchLightning/pytorch-lightning/blob/
# 80c529351439a0f8d3d6e9449cd47d16ba3abbec/pytorch_lightning/
# callbacks/early_stopping.py
class EarlyStoppingMinEpochs(pl.callbacks.Callback):
    def __init__(
        self,
        monitor: str = "early_stop_on",
        min_epochs: int = 500,
        patience: int = 200,
    ):
        super(EarlyStoppingMinEpochs, self).__init__()
        self.monitor = monitor
        self.min_epochs = min_epochs
        self.patience = patience
        self.best_epoch = None
        self.best_value = None
        self.monitor_log = []

    def _validate_condition_metric(self, logs):
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"Early stopping conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `EarlyStoppingMinEpochs` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            raise RuntimeError(error_msg)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):

        if trainer.current_epoch <= self.min_epochs:
            # do not track best epoch here just yet
            return

        logs = trainer.callback_metrics
        self._validate_condition_metric(logs)

        current = logs.get(self.monitor).item()
        if self.best_epoch is None or current < self.best_value:
            self.best_value = current
            self.best_epoch = trainer.current_epoch

        trainer.should_stop = trainer.current_epoch > self.best_epoch + self.patience
