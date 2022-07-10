from datetime import datetime
from copy import deepcopy
from functools import cache
from math import floor
from io import StringIO
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from os import rename
from pathlib import Path
import time
import sys
import warnings

from monty.json import MSONable
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from xas_nne.ml import _OptimizerSetter, _activation_map, \
    FeedForwardNeuralNetwork


T1 = 0.39908993417  # 0.5 ln (2 pi)


class LightningMVE(_OptimizerSetter, pl.LightningModule):
    def __init__(
        self,
        *,
        input_size,
        hidden_sizes,
        output_size,
        dropout=0.0,
        batch_norm=True,
        last_batch_norm=False,
        activation="relu",
        last_activation=None,
        print_every_epoch=50,
    ):
        super().__init__()
        self.save_hyperparameters()

        activation = _activation_map(activation)
        last_activation = _activation_map(last_activation)

        # The output size of the NNE should be 2x the actual output size since
        # we're predicting both the mean and log-variance
        self._model = FeedForwardNeuralNetwork(
            architecture=[input_size, *hidden_sizes, int(2*output_size)],
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            last_activation=last_activation,
            last_batch_norm=last_batch_norm,
        )

        self._print_every_epoch = print_every_epoch
        self._epoch_dt = 0.0
        self._output_size = output_size

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions

        return self._model(x)

    @staticmethod
    def _nnl_loss(y, mu, log_variance):
        variance_contribution = (0.5 * log_variance).mean()
        combined = (torch.abs(y - mu) / 2.0 * torch.exp(-log_variance)).mean()
        return {"s2_loss": variance_contribution, "scaled_mse_loss": combined}

    def _single_forward_step(self, batch, batch_index):
        """Executes a single forward pass given some batch and batch index."""

        (x, y) = batch
        y_hat = self(x)
        mu = y_hat[:, :self._output_size]
        log_variance = y_hat[:, self._output_size:]
        return self._nnl_loss(y, mu, log_variance)

    def training_step(self, batch, batch_idx):
        losses = self._single_forward_step(batch, batch_idx)
        return {
            "loss": (T1 + losses["s2_loss"] + losses["scaled_mse_loss"]).mean(),
            "s2_loss": losses["s2_loss"].mean().item(),
            "scaled_mse_loss": losses["scaled_mse_loss"].mean().item()
        }

    def validation_step(self, batch, batch_idx):
        losses = self._single_forward_step(batch, batch_idx)
        return {
            "loss": (T1 + losses["s2_loss"] + losses["scaled_mse_loss"]).mean(),
            "s2_loss": losses["s2_loss"].mean().item(),
            "scaled_mse_loss": losses["scaled_mse_loss"].mean().item()
        }

    def on_train_epoch_start(self):
        self._epoch_dt = time.time()

    def _log_outputs(
        self,
        outputs,
        what="train",
        keys=["loss", "s2_loss", "scaled_mse_loss"]
    ):

        d = {}
        for key in keys:
            tmp_loss = torch.tensor([x[key] for x in outputs]).mean().item()
            d[key] = tmp_loss
            self.log(f"{what}_{key}", tmp_loss, on_step=False, on_epoch=True)
        return d

    def training_epoch_end(self, outputs):
        d = self._log_outputs(outputs, "train")
        epoch = self.trainer.current_epoch + 1
        dt = time.time() - self._epoch_dt
        if self._print_every_epoch > 0:
            if epoch % self._print_every_epoch == 0:
                loss = d["loss"]
                s2 = d["s2_loss"]
                smse = d["scaled_mse_loss"]
                print(f"\ttr loss {loss:.03e} | {(dt/60.0):.02f} m")
                print(f"\ts2 {s2:.03e} | scaled_mse {smse:.03e}", flush=True)

    def validation_epoch_end(self, outputs):
        d = self._log_outputs(outputs, "val")
        epoch = self.trainer.current_epoch + 1

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True)

        if self._print_every_epoch > 0:
            if epoch % self._print_every_epoch == 0:
                loss = d["loss"]
                s2 = d["s2_loss"]
                smse = d["scaled_mse_loss"]
                print(f"Epoch {epoch:05}")
                print(f"\tlr: {lr:.03e}")
                print(f"\tcv loss {loss:.03e}")
                print(f"\ts2 {s2:.03e} | scaled_mse {smse:.03e}", flush=True)

    def on_train_end(self):
        """Logs information as training ends."""

        if self.trainer.global_rank == 0:
            epoch = self.trainer.current_epoch + 1
            if epoch < self.trainer.max_epochs:
                print(
                    "Early stopping criteria reached at "
                    f"epoch {epoch}/{self.trainer.max_epochs}"
                )
