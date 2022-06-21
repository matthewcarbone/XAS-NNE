"""Core machine learning module. Implements various helper classes for running
a simple feed-forward neural network ensemble, using Pytorch and Pytorch
Lightning as the back-ends."""

from pathlib import Path
import time

import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl


def _activation_map(s):
    if s == "relu":
        return nn.ReLU()
    elif s == "sigmoid":
        return nn.Sigmoid()
    elif s is None:
        return None
    else:
        raise ValueError(s)


class FeedforwardLayer(nn.Module):
    def __init__(
        self,
        *,
        input_size,
        output_size,
        activation=nn.ReLU(),
        dropout=0.0,
        batch_norm=False,
    ):
        super().__init__()
        layers = [torch.nn.Linear(input_size, output_size)]
        if activation is not None:
            layers.append(activation)
        if batch_norm:
            layers.append(nn.BatchNorm1d(output_size))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(
        self,
        *,
        architecture,
        dropout=0.0,
        activation=nn.ReLU(),
        last_activation=None,
        batch_norm=False,
        last_batch_norm=False,
    ):

        super().__init__()
        assert len(architecture) > 1

        layers = []
        for ii, (n, n2) in enumerate(zip(architecture[:-1], architecture[1:])):
            if ii == len(architecture) - 2:
                a = last_activation
                b = last_batch_norm
            else:
                a = activation
                b = batch_norm
            layers.append(
                FeedforwardLayer(
                    input_size=n,
                    output_size=n2,
                    activation=a,
                    dropout=dropout,
                    batch_norm=b,
                )
            )

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        return self._layers(x)


class _OptimizerSetter:
    def set_optimizer(self, optimizer, scheduler, scheduler_kwargs):
        """Flexible method for setting the optimizer and scheduler parameters
        of the model. This configures the ``_optimizer_arguments`` attributes,
        which are then used in the ``configure_optimizers`` method.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        scheduler : torch.optim.Scheduler
        scheduler_kwargs : dict
            Optional arguments to pass to the pytorch lightning API for
            monitoring the scheduler. For example, ``{"monitor": "val_loss"}``.
        """

        if scheduler is None:
            self._optimizer_arguments = {"optimizer": optimizer}
        self._optimizer_arguments = {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, **scheduler_kwargs},
        }

    def configure_optimizers(self):
        """Core method for pytorch lightning. Returns the
        ``_optimizer_arguments`` attribute.

        Returns
        -------
        dict
        """

        return self._optimizer_arguments


class _GeneralPLModule:
    def training_step(self, batch, batch_idx):
        loss = self._single_forward_step(batch, batch_idx)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._single_forward_step(batch, batch_idx)
        return {"loss": loss}

    def on_train_epoch_start(self):
        self._epoch_dt = time.time()

    def _log_outputs(self, outputs, what="train", keys=["loss"]):
        d = {}
        for key in keys:
            tmp_loss = torch.tensor([x[key] for x in outputs]).mean().item()
            d[key] = tmp_loss
            self.log(f"{what}_{key}", tmp_loss, on_step=False, on_epoch=True)
        return d

    def _single_forward_step(self, batch, batch_index):
        raise NotImplementedError

    def training_epoch_end(self, outputs):
        d = self._log_outputs(outputs, "train")
        epoch = self.trainer.current_epoch + 1
        dt = time.time() - self._epoch_dt
        if self._print_every_epoch > 0:
            if epoch % self._print_every_epoch == 0:
                loss = d["loss"]
                print(
                    f"\ttr loss {loss:.03e} | {(dt/60.0):.02f} m", flush=True
                )

    def validation_epoch_end(self, outputs):
        d = self._log_outputs(outputs, "val")
        epoch = self.trainer.current_epoch + 1

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True)

        if self._print_every_epoch > 0:
            if epoch % self._print_every_epoch == 0:
                loss = d["loss"]
                print(f"Epoch {epoch:05}")
                print(f"\tlr: {lr:.03e}")
                print(f"\tcv loss {loss:.03e}")

    def on_train_end(self):
        """Logs information as training ends."""

        if self.trainer.global_rank == 0:
            epoch = self.trainer.current_epoch + 1
            if epoch < self.trainer.max_epochs:
                print(
                    "Early stopping criteria reached at "
                    f"epoch {epoch}/{self.trainer.max_epochs}"
                )


class LightningMultiLayerPerceptron(
    _OptimizerSetter, _GeneralPLModule, pl.LightningModule
):
    def __init__(
        self,
        *,
        input_size,
        hidden_sizes,
        output_size,
        dropout=0.0,
        batch_norm=True,
        activation="relu",
        before_latent_activation="sigmoid",  # encoder output
        criterion="mse",
        last_batch_norm=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        activation = _activation_map(activation)
        before_latent_activation = _activation_map(before_latent_activation)

        self._model = FeedForwardNeuralNetwork(
            architecture=[input_size, *hidden_sizes, output_size],
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            last_activation=before_latent_activation,
            last_batch_norm=last_batch_norm,
        )

        if criterion == "mse":
            self.criterion = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        self._print_every_epoch = 0
        self._epoch_dt = 0.0

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions

        return self._model(x)

    def _single_forward_step(self, batch, batch_index):
        """Executes a single forward pass given some batch and batch index.
        In this model, we first encode using self(x)"""

        (x, y) = batch
        y_hat = self(x)
        mse_loss = self.criterion(y_hat, y)  # reduction = mean already applies
        return mse_loss


class Trainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export_csv_log(
        self, columns=["epoch", "train_loss", "val_loss", "lr"]
    ):
        """Custom method for exporting the trainer logs to something much more
        readable. Only executes on the 0th global rank for DDP jobs."""

        if self.global_rank > 0:
            return

        metrics = self.logger.experiment.metrics
        log_dir = self.logger.experiment.log_dir

        path = Path(log_dir) / Path("custom_metrics.csv")
        t = pd.DataFrame([d for d in metrics if "train_loss" in d])
        v = pd.DataFrame([d for d in metrics if "val_loss" in d])
        df = pd.concat([t, v], join="outer", axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        try:
            df = df[columns]
            df.to_csv(path, index=False)
        except KeyError:
            print(
                "might be running overfit_batches, not saving custom metrics"
            )

    @staticmethod
    def get_best_model_checkpoint_path(path):
        """Returns the checkpoint path corresponding to the best model.

        Parameters
        ----------
        path : os.PathLike
            The path to the directory containing the checkpoint files.

        Raises
        ------
        FileNotFoundError
            If checkpoint files are missing, or there is any issue with the
            directory provided.

        Returns
        -------
        str
            The best checkpoint file so far.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Directory {path} does not exist")

        files = list(path.iterdir())
        if len(files) == 0:
            raise FileNotFoundError(f"Directory {path} is empty")

        extensions = [str(f).split(".")[1] == "ckpt" for f in files]
        if not any(extensions):
            raise FileNotFoundError(f"Directory {path} has no ckpt files")

        checkpoints = list(path.rglob("*.ckpt"))
        checkpoints = [
            [cc, int(str(cc).split("=")[1].split("-")[0])]
            for cc in checkpoints
        ]
        checkpoints.sort(key=lambda x: x[1])
        checkpoints = [str(cc[0]) for cc in checkpoints]
        return str(checkpoints[-1])

    def fit(self, **kwargs):
        print_every_epoch = 0
        if "print_every_epoch" in kwargs.keys():
            print_every_epoch = kwargs.pop("print_every_epoch")
        kwargs["model"]._print_every_epoch = print_every_epoch
        super().fit(**kwargs)
        self.export_csv_log()


class Ensemble:

    def __init__(self, path, models=[]):
        self._path = Path(path)
        self._models = models

