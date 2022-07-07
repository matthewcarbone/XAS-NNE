import numpy as np
from sklearn.preprocessing import StandardScaler


def select_data_small_molecule_MD(
    training_indexes,
    loaded_data,
    test_index_start=None,
    scale_x=True,
    scale_y=True
):
    """Takes a dictionary with keys
    ``dict_keys(['grid', 'y', 'x', 'snapshots', 'sites'])`` and returns another
    dictionary with the same keys, but split into testing and training. This
    function does this in the following way. First, the training data is
    selected from ``training_indexes``. Then, the testing data is selected
    starting from ``test_index_start`` onwards. This means that some data
    can be omitted during parsing, but this is by design. The intent is to use
    this function to help generate data for active learning, where only a
    subset of the overall possible data is used for training.


    Parameters
    ----------
    training_indexes : array_like
    loaded_data : dict
        Should contain the keys "grid", "y", "x", "snapshots", "sites".
    test_index_start : int, optional
        If None, selects the maximum training set index.
    scale_x : bool, optional
        If True, uses a StandardScaler to scale the x input, fitting on the
        training data.
    scale_y : bool, optional
        Same as above but for the target data.

    Returns
    -------
    dict, StandardScaler, StandardScaler
    """

    loaded_data = loaded_data.copy()
    
    total_datapoints = loaded_data["x"].shape[0]
    all_points = set([ii for ii in range(total_datapoints)])
    all_other_indexes = list(all_points - set(training_indexes))
    assert set(training_indexes).isdisjoint(set(all_other_indexes))
    if test_index_start is None:
        test_index_start = max(training_indexes) + 1
    assert test_index_start >= max(training_indexes)
    
    data = dict(
        train=dict(
            x=loaded_data["x"][training_indexes, :].copy(),
            y=loaded_data["y"][training_indexes, :].copy(),
            grid=loaded_data["grid"],
            snapshots=[
                loaded_data["snapshots"][ii] for ii in training_indexes
            ],
            sites=[loaded_data["sites"][ii] for ii in training_indexes],
        ),
        test=dict(
            x=loaded_data["x"][test_index_start:, :].copy(),
            y=loaded_data["y"][test_index_start:, :].copy(),
            grid=loaded_data["grid"],
            snapshots=[
                xx for xx in loaded_data["snapshots"][test_index_start:]
            ],
            sites=[xx for xx in loaded_data["sites"][test_index_start:]],
        ),
    )
    
    x_scaler = None
    if scale_x:
        x_scaler = StandardScaler()
        data["train"]["x"] = x_scaler.fit_transform(data["train"]["x"])
        data["test"]["x"] = x_scaler.transform(data["test"]["x"])
        
    y_scaler = None
    if scale_y:    
        y_scaler = StandardScaler()
        data["train"]["y"] = y_scaler.fit_transform(data["train"]["y"])
        data["test"]["y"] = y_scaler.transform(data["test"]["y"])
        
    return data, x_scaler, y_scaler


def get_predictions(data, ensemble, y_scaler=None):
    """Gets the predictions of an ensemble. Note that the "x" data contained in
    ``data`` should be exactly as the estimators in the ensemble expect it.
    This function will also scale the output data properly if a scaler is
    provided.

    Parameters
    ----------
    data : dict
        Should have at least the keys "x" and "y".
    ensemble : xas_nne.ml.Ensemble
    y_scaler : StandardScaler, optional
    """

    predictions = ensemble.predict(data["x"])
    ground_truth = data["y"]

    # If y_scaler is provided, unscale
    if y_scaler is not None:
        N_ensembles = predictions.shape[0]
        M = predictions.shape[-1]
        predictions = y_scaler.inverse_transform(
            predictions.reshape(-1, M)
        ).reshape(N_ensembles, -1, M)
        ground_truth = y_scaler.inverse_transform(ground_truth)

    return predictions, ground_truth


def get_molecular_spectra_MD(
    data, ensemble, atoms_per_molecule=6, y_scaler=None
):
    """Combines data intelligently based on the number of atoms/molecule.
    Note that this requires the data to be properly ordered. The provided
    data dictionary should also contain the "snapshots" key in addition to "x"
    and "y"."""

    pred, gt = get_predictions(data, ensemble, y_scaler)

    pred = pred.reshape(pred.shape[0], -1, atoms_per_molecule, pred.shape[-1])
    gt = gt.reshape(-1, atoms_per_molecule, gt.shape[-1])
    snapshots = data["snapshots"][::atoms_per_molecule]

    molecular_gt = gt.mean(axis=1)
    molecular_preds = pred.mean(axis=0).mean(axis=1)
    molecular_spreads = np.sqrt(
        (pred.std(axis=0)**2).sum(axis=1)
    ) / atoms_per_molecule

    return snapshots, molecular_gt, molecular_preds, molecular_spreads
