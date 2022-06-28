import numpy as np

from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


def random_downsample(arrays, keep_prop=0.9, replace=False, seed=None):
    """Takes an arbitrary number of arrays as input arguments and returns
    randomly downsampled versions.

    Parameters
    ----------
    arrays
        A list of arrays to downsample, all in the same way
    size : None, optional
        Description
    replace : bool, optional
        Description
    p : None, optional
        Description
    seed : None, optional
        Description

    Returns
    -------
    TYPE
        Description
    """

    np.random.seed(seed)

    if keep_prop == 1.0:
        return arrays
    elif keep_prop > 1.0 or keep_prop < 0.0:
        raise ValueError(f"keep_prop {keep_prop} must be in 0, 1")
    L = arrays[0].shape[0]  # Number of examples
    choice = np.random.choice(
        L,
        size=int(keep_prop * L),
        replace=replace,
        p=None
    )
    return [arr[choice, ...] for arr in arrays]


class Data(pl.LightningDataModule):
    """Base Data class inheriting ``pl.LightningDataModule``. Used for simple
    data on disk, nothing fancy."""

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_targets(self):
        return self._n_features

    def __init__(
        self,
        *,
        train,
        val,
        train_loader_kwargs={
            "batch_size": 64,
            "persistent_workers": True,
            "pin_memory": True,
            "num_workers": 3,
        },
        val_loader_kwargs={
            "batch_size": 64,
            "persistent_workers": True,
            "pin_memory": True,
            "num_workers": 3,
        },
        downsample_training_proportion=1.0,
        parallel=False,
    ):
        super().__init__()
        if set(list(train.keys())) != set(list(val.keys())):
            raise ValueError("Keys must be the same for train and val")
        if "x" not in train.keys():
            raise ValueError("Keys must contain x")

        # Possibly unsupervised only
        _train = [train["x"]]
        _val = [val["x"]]
        self._n_features = train["x"].shape[1]
        assert self._n_features == val["x"].shape[1]

        # Supervised
        if "y" in train.keys():
            _train.append(train["y"])
            _val.append(val["y"])
            self._n_targets = train["y"].shape[1]
            assert self._n_targets == val["y"].shape[1]
        else:
            self._n_targets = self._n_features

        # Down-sample the training data if necessary. This is useful for
        # ensembling and training on different subsets of the database.
        if downsample_training_proportion < 1.0:
            _train = random_downsample(
                _train,
                keep_prop=downsample_training_proportion,
                replace=False,
            )

        if parallel:
            # self._train_loader_kwargs["num_workers"] = 1
            self._train_loader_kwargs["multiprocessing_context"] = 'fork'
            # self._val_loader_kwargs["num_workers"] = 1
            self._val_loader_kwargs["multiprocessing_context"] = 'fork'

        self._train_data = (Tensor(xx) for xx in _train)
        self._val_data = (Tensor(xx) for xx in _val)
        self._train_loader_kwargs = train_loader_kwargs
        self._val_loader_kwargs = val_loader_kwargs

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(*self._train_data), **self._train_loader_kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(*self._val_data), **self._val_loader_kwargs
        )
