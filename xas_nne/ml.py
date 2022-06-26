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

from exafs_tools.data import Data


def _activation_map(s):
    if s == "relu":
        return nn.ReLU()
    elif s == "sigmoid":
        return nn.Sigmoid()
    elif s == "softplus":
        return nn.Softplus()
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

    @classmethod
    def from_random_architecture(
        cls,
        input_size,
        min_layers=3,
        max_layers=7,
        min_neurons_per_layer=80,
        max_neurons_per_layer=120,
        dropout=0.0,
        batch_norm=False,
        activation="relu",
        last_activation=None,
        criterion="mse",
        last_batch_norm=False,
        seed=None,
    ):

        np.random.seed(seed)
        n_hidden_layers = np.random.randint(
            low=min_layers,
            high=max_layers + 1
        )
        architecture = np.random.randint(
            low=min_neurons_per_layer,
            high=max_neurons_per_layer,
            size=(n_hidden_layers,)
        )
        return LightningMultiLayerPerceptron(
            input_size=input_size,
            hidden_sizes=architecture[:-1],
            output_size=architecture[-1],
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            last_activation=last_activation,
            criterion=criterion,
            last_batch_norm=last_batch_norm,
        )

    def __init__(
        self,
        *,
        input_size,
        hidden_sizes,
        output_size,
        dropout=0.0,
        batch_norm=True,
        activation="relu",
        last_activation="softplus",
        criterion="mse",
        last_batch_norm=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        activation = _activation_map(activation)
        last_activation = _activation_map(last_activation)

        self._model = FeedForwardNeuralNetwork(
            architecture=[input_size, *hidden_sizes, output_size],
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            last_activation=last_activation,
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


@cache
def load_LightningMultiLayerPerceptron_from_ckpt(path):
    """Loads the LightningMultiLayerPerceptron from path, but the results are
    cached so as to speed up the loading process dramatically.

    Parameters
    ----------
    path : os.PathLike

    Returns
    -------
    LightningMultiLayerPerceptron
    """

    return LightningMultiLayerPerceptron.load_from_checkpoint(path)


class SingleEstimator(MSONable):

    def get_default_logger(self):
        return CSVLogger(self._root, name="Logs")

    def get_default_early_stopper(self, **kwargs):
        """Gets the default early stopper. Some of the defaults should be
        something like

        monitor="val_loss"
        check_finite=True
        patience=100
        verbose=False
        """

        return EarlyStopping(monitor="train_loss", **kwargs)

    def get_trainer(
        self,
        max_epochs=100,
        monitor="val_loss",
        early_stopper_patience=100,
        gpus=None
    ):
        """Initializes and returns the trainer object.

        Parameters
        ----------
        max_epochs : int, optional
        monitor : str, optional
        early_stopper_patience : int, optional

        Returns
        -------
        Trainer
        """

        logger = self.get_default_logger()
        early_stopper = self.get_default_early_stopper(
            monitor=monitor,
            check_finite=True,
            patience=early_stopper_patience,
            verbose=False
        )
        cuda = torch.cuda.is_available()
        checkpointer = ModelCheckpoint(
            dirpath=f"{self._root}/Checkpoints",
            save_top_k=5,
            monitor=monitor
        )
        print(f"Setting trainer with cuda={cuda}")

        if gpus is None:
            gpus = int(cuda)
            auto_select_gpus = bool(cuda)
        else:
            assert isinstance(gpus, int)
            auto_select_gpus = True
        return Trainer(
            gpus=gpus,
            num_nodes=1,
            auto_select_gpus=auto_select_gpus,
            precision=32,
            max_epochs=max_epochs,
            enable_progress_bar=False,
            log_every_n_steps=1,
            logger=logger,
            callbacks=[early_stopper, checkpointer],
            enable_model_summary=True,
        )

    @staticmethod
    def set_optimizer_family(
        model,
        lr=1e-2,
        patience=10,
        min_lr=1e-7,
        factor=0.95,
        monitor="val_loss",
    ):
        local = {
            key: value for key, value in locals().items() if key != "model"
        }
        print(f"Setting optimizer family: {local}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=patience,
            min_lr=min_lr,
            factor=factor,
        )
        scheduler_kwargs = {"monitor": monitor}
        model.set_optimizer(optimizer, scheduler, scheduler_kwargs)

    def _set_root(self, root):
        if root is not None:
            self._root = str(root)
            Path(self._root).mkdir(exist_ok=True, parents=True)
        else:
            self._root = None

    @property
    def best_checkpoint(self):
        return self._best_checkpoint

    @property
    def best_model(self):
        return load_LightningMultiLayerPerceptron_from_ckpt(
            self._best_checkpoint
        )

    def __init__(
        self,
        root=None,
        from_random_architecture_kwargs={
            "min_layers": 3,
            "max_layers": 7,
            "min_neurons_per_layer": 150,
            "max_neurons_per_layer": 200,
            "dropout": 0.0,
            "batch_norm": False,
            "activation": "relu",
            "criterion": "mse",
            "last_activation": None,
            "last_batch_norm": False,
        },
        best_checkpoint=None,
        last_lr=None
    ):
        self._set_root(root)
        self._best_checkpoint = best_checkpoint
        self._last_lr = last_lr
        self._from_random_architecture_kwargs = from_random_architecture_kwargs

    def train(
        self,
        *,
        training_data,
        model=None,
        override_root=None,
        checkpoint=None,
        val_prop=0.1,
        batch_size=4096,
        persistent_workers=True,
        pin_memory=True,
        num_workers=3,
        epochs=100,
        lr=None,
        patience=10,
        min_lr=1e-7,
        factor=0.95,
        monitor="val_loss",
        early_stopper_patience=100,
        gpus=None,
        seed=None,
    ):
        """Trains a model. If model is None, will attempt to load one from
        state.

        Parameters
        ----------
        training_data : numpy.array
            Description
        model : None, optional
            Description
        checkpoint : None, optional
            Description
        val_prop : float, optional
            Description
        batch_size : int, optional
            Description
        persistent_workers : bool, optional
            Description
        pin_memory : bool, optional
            Description
        num_workers : int, optional
            Description
        epochs : int, optional
            Description
        override_root : None, optional
            Description
        lr : float, optional
            Description
        patience : int, optional
            Description
        min_lr : float, optional
            Description
        factor : float, optional
            Description
        monitor : str, optional
            Description
        early_stopper_patience : int, optional
            Description
        gpus : None, optional
            Description
        seed : None, optional
            Description
        """

        if seed is not None:
            seed_everything(seed)

        if model is None:
            if checkpoint is not None:
                model = LightningMultiLayerPerceptron.load_from_checkpoint(
                    checkpoint
                )
                print(f"Reloaded model from provided {checkpoint}")
            elif self._best_checkpoint is not None:
                model = self.best_model
                print(
                    "Initialized model from best stored model at "
                    f"{self._best_checkpoint}"
                )
            else:
                model = LightningMultiLayerPerceptron.from_random_architecture(
                    training_data.shape[1],
                    **self._from_random_architecture_kwargs
                )
                print(
                    "Initialized model from random architecture using "
                    f"arguments {self._from_random_architecture_kwargs}"
                )

        if lr is None:
            if self._last_lr is not None:
                lr = self._last_lr
                print(f"Learning rate loaded from stored: {lr}")
            else:
                lr = 1e-2  # Good default

        # After loading the data from root, we can override it to save to a
        # new location
        if override_root is not None:
            print(f"Root set to {override_root}")
            self._set_root(override_root)

        # Execute the training using a lot of defaults/boilerplate
        self.set_optimizer_family(
            model,
            lr=lr,
            patience=patience,
            min_lr=min_lr,
            factor=factor,
            monitor=monitor,
        )

        # Trainer
        trainer = self.get_trainer(
            max_epochs=epochs,
            early_stopper_patience=early_stopper_patience,
            gpus=gpus,
            monitor=monitor,
        )

        # Loader
        L = len(training_data["x"])
        val_number = int(L * val_prop)
        train_number = L - val_number
        train_idx, val_idx = random_split(
            range(L),
            [train_number, val_number],
            generator=torch.Generator().manual_seed(seed) if seed is not None
            else None
        )
        _train_data = {
            "x": training_data["x"][train_idx, :],
            "y": training_data["y"][train_idx, :]
        }
        _val_data = {
            "x": training_data["x"][val_idx, :],
            "y": training_data["y"][val_idx, :]
        }
        loader = Data(
            train=_train_data,
            val=_val_data,
            train_loader_kwargs={
                "batch_size": batch_size,
                "persistent_workers": persistent_workers,
                "pin_memory": pin_memory,
                "num_workers": num_workers,
            },
        )

        # Execute training
        trainer.fit(
            model=model,
            train_dataloaders=loader,
            print_every_epoch=epochs // 5
        )
        self._best_checkpoint = trainer.checkpoint_callback.best_model_path
        self._last_lr = trainer.optimizers[0].param_groups[0]["lr"]

    def predict(self, x, model=None):
        """Makes a prediction on the provided data.

        Parameters
        ----------
        x : numpy.array

        Returns
        -------
        numpy.array
        """

        if model is None:
            model = self.best_model

        x = torch.Tensor(x)
        with torch.no_grad():
            model.eval()
            return model.forward(x).detach().numpy()


# class CaptureOutput():
#     def __enter__(self):
#         self.record = {"stdout": None, "stderr": None}
#         self._stdout = sys.stdout
#         self._stderr = sys.stderr
#         sys.stdout = self._mystdout = StringIO()
#         sys.stderr = self._mystderr = StringIO()
#         return self

#     def __exit__(self, *args):
#         self.record["stdout"] = self._mystdout.getvalue().splitlines()
#         self.record["stderr"] = self._mystderr.getvalue().splitlines()
#         sys.stdout = self._stdout
#         sys.stderr = self._stderr


# class IdentityEnsemble(MSONable):

#     @classmethod
#     def from_random_architectures(
#         cls,
#         root,
#         n_estimators=10,
#         from_random_architecture_kwargs={
#             "min_layers": 2,
#             "max_layers": 4,
#             "min_neurons_per_layer": 80,
#             "max_neurons_per_layer": 120,
#             "dropout": 0.0,
#             "batch_norm": False,
#             "activation": "relu",
#             "before_latent_activation": None,
#             "criterion": "mse",
#             "last_activation": None,
#         },
#         seed=None
#     ):
#         if Path(root).exists():
#             now = datetime.now().strftime("%y%m%d-%H%M%S")
#             old_root = str(root) + f"-{now}"
#             rename(str(root), old_root)
#             print(f"Renamed existing root {root} to {old_root}")
#         if seed is not None:
#             seed_everything(seed)
#         estimators = [
#             SingleLightningAutoencoderEstimator(
#                 from_random_architecture_kwargs=from_random_architecture_kwargs
#             )
#             for _ in range(n_estimators)
#         ]
#         return cls(root, estimators)

#     def __init__(self, root, estimators):
#         self._root = str(root)
#         self._estimators = estimators

#     def _get_ensemble_model_root(self, ensemble_index, estimator_index):
#         return Path(self._root) / Path(f"{ensemble_index:06}") / \
#             Path(f"{estimator_index:06}")

#     def train(
#         self,
#         training_data,
#         ensemble_index=0,
#         estimator_index=0,
#         epochs=100,
#         lr=None
#     ):
#         print(f"Training estimator {estimator_index}")
#         estimator = self._estimators[estimator_index]
#         estimator.train(
#             training_data,
#             epochs=epochs,
#             override_root=self._get_ensemble_model_root(
#                 ensemble_index, estimator_index
#             ),
#             lr=lr
#         )

#     def train_ensemble_serial(
#         self,
#         training_data,
#         ensemble_index=0,
#         epochs=100,
#         lr=None
#     ):
#         """Trains the entire ensemble in serial. Default behavior is to
#         reload existing models from checkpoint and to train them using the
#         provided learning rate, and other parameters.

#         Parameters
#         ----------
#         training_data : TYPE
#             Description
#         ensemble_index : int, optional
#             Description
#         epochs : int, optional
#             Description
#         lr : None, optional
#             Description
#         """

#         for ii in range(len(self._estimators)):
#             self.train(
#                 training_data,
#                 ensemble_index=ensemble_index,
#                 estimator_index=ii,
#                 epochs=epochs,
#                 lr=lr
#             )

#     def predict(self, x):
#         """Predicts on the provided data in ``x`` by loading the best models
#         from disk.

#         Parameters
#         ----------
#         x : numpy.array
#         """

#         results = []
#         for estimator in self._estimators:
#             results.append(estimator.predict(x))
#         return np.array(results)

#     def train_ensemble_parallel(
#         self,
#         training_data,
#         ensemble_index=0,
#         epochs=100,
#         lr=None,
#         n_jobs=cpu_count() // 2
#     ):

#         warnings.warn(
#             "This is highly experimental! Recommended to just train "
#             "in serial for now"
#         )

#         def _run_wrapper(estimator_index, estimator):
#             estimator.train(
#                 training_data,
#                 epochs=epochs,
#                 override_root=self._get_ensemble_model_root(
#                     ensemble_index, estimator_index
#                 ),
#                 lr=lr,
#                 parallel=True
#             )
#             print(
#                 "Trained ensemble/estimator "
#                 f"{ensemble_index}/{estimator_index}",
#                 flush=True
#             )
#             return deepcopy(estimator)

#         results = Parallel(n_jobs=n_jobs)(
#             delayed(_run_wrapper)(ii, estimator)
#             for ii, estimator in enumerate(self._estimators)
#         )

#         self._estimators = results
