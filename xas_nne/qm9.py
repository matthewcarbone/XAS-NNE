"""General utilities for parsing the QM9 database."""

from collections import Counter
from functools import cache

import numpy as np
from rdkit import Chem
import torch
from tqdm import tqdm


def parse_QM9_scalar_properties(props, selected_properties=None):
    """Parses a list of strings into the correct floats that correspond to the
    molecular properties in the QM9 database.

    Only the following properties turn out to be statistically relevant in this
    dataset: selected_properties=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 14]. These
    properties are the statistically independent contributions as calculated
    via a linear correlation model, and together the capture >99% of the
    variance of the dataset.

    Note, according to the paper (Table 3)
    https://www.nature.com/articles/sdata201422.pdf
    the properties are as follows (1-indexed):
        1  : gdb9 index (we'll ignore)
        2  : identifier
        3  : "A" (GHz) rotational constant
        4  : "B" (GHz) rotational constant
        5  : "C" (GHz) rotational constant
        6  : "mu" (Debeye) dipole moment
        7  : "alpha" (a0^3) isotropic polarizability
        8  : "HOMO energy" (Ha)
        9  : "LUMO energy" (Ha)
        10 : "E gap" (Ha) 8-9 (might be uHa?)
        11 : "<R^2>" (a0^2) electronic spatial extent
        12 : "zpve" (Ha) zero-point vibrational energy
        13 : "U0" (Ha) internal energy at 0 K
        14 : "U" (Ha) internal energy at 198.15 K
        15 : "H" (Ha) enthalpy at 298.15 K
        16 : "G" (Ha) gibbs free energy at 298.15 K
        17 : "Cv" (cal/molK) heat capacity at 298.15 K

    The relevant ones (2 through 17 inclusive) will be returned in a new list
    with each element being of the correct type.

    Parameters
    ----------
    props : list of str
        Initial properties in string format.
    selected_properties : List[int], optional
        Selected properties. If None, take all the properties.

    Returns
    -------
    tuple
        The QM9 ID and list of properties (list of float).
    """

    qm9_id = int(props[1])
    other = props[2:]

    if selected_properties is None:
        other = [float(prop) for ii, prop in enumerate(other)]
    else:
        other = [
            float(prop)
            for ii, prop in enumerate(other)
            if ii in selected_properties
        ]
    return (qm9_id, other)


def read_qm9_xyz(xyz_path):
    """Function for reading .xyz files like those present in QM9. Note this
    does not read in geometry information, just properties and SMILES codes.
    For a detailed description of the properties contained in the qm9 database,
    see this manuscript: https://www.nature.com/articles/sdata201422.pdf

    Parameters
    ----------
    xyz_path : str
        Absolute path to the xyz file.

    Returns
    -------
    dict
        Dictionary containing all information about the loaded data point.
    """

    with open(xyz_path, "r") as file:
        n_atoms = int(file.readline())
        qm9id, other_props = parse_QM9_scalar_properties(
            file.readline().split()
        )

        elements = []
        xyzs = []
        for ii in range(n_atoms):
            line = file.readline().replace(".*^", "e").replace("*^", "e")
            line = line.split()
            elements.append(str(line[0]))
            xyzs.append(np.array(line[1:4], dtype=float))

        xyzs = np.array(xyzs)

        # Skip extra vibrational information
        file.readline()

        # Now read the SMILES code
        smiles = file.readline().split()
        _smiles = smiles[0]
        _canon_smiles = smiles[1]

        zwitter = "+" in smiles[0] or "-" in smiles[0]

    return {
        "qm9id": qm9id,
        "smiles": _smiles,
        "canon_smiles": _canon_smiles,
        "other": other_props,
        "xyz": xyzs.tolist(),
        "elements": elements,
        "zwitter": zwitter,
    }


def remove_zwitter_ions_(data):
    """Removes molecules that are classified as Zwitter-ionic. Operates
    in-place.

    Parameters
    ----------
    data : dict
    """

    where_not_zwitter = np.where(np.array([
        "+" not in smi and "-" not in smi for smi in data["origin_smiles"]
    ]) == 1)[0]
    data["x"] = data["x"][where_not_zwitter, :]
    data["y"] = data["y"][where_not_zwitter, :]
    data["names"] = [data["names"][ii] for ii in where_not_zwitter]
    data["origin_smiles"] = [
        data["origin_smiles"][ii] for ii in where_not_zwitter
    ]
    L = len(data["x"])
    print(f"Down-sampled to {L} data after removing zwitter ions")


@cache
def atom_count(smile):
    """Gets the atom count, resolved by atom type. Does not count hydrogen.

    Parameters
    ----------
    smile : str

    Returns
    -------
    dict
    """

    mol = Chem.MolFromSmiles(smile)
    return dict(Counter([atom.GetSymbol() for atom in mol.GetAtoms()]))


def _qm9_train_val_test_from_data(data, where_train, where_val, where_test):
    """Summary

    Parameters
    ----------
    data : TYPE
        Description
    where_train : TYPE
        Description
    where_val : TYPE
        Description
    where_test : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """

    assert set(where_train).isdisjoint(set(where_val))
    assert set(where_train).isdisjoint(set(where_test))
    assert set(where_val).isdisjoint(set(where_test))

    train = {
        "grid": data["grid"],
        "x": data["x"][where_train, :],
        "y": data["y"][where_train, :],
        "names": [data["names"][ii] for ii in where_train],
        "origin_smiles": [data["origin_smiles"][ii] for ii in where_train],
        "n_atoms": [
            atom_count(data["origin_smiles"][ii]) for ii in where_train
        ],
    }
    val = {
        "grid": data["grid"],
        "x": data["x"][where_val, :],
        "y": data["y"][where_val, :],
        "names": [data["names"][ii] for ii in where_val],
        "origin_smiles": [data["origin_smiles"][ii] for ii in where_val],
        "n_atoms": [
            atom_count(data["origin_smiles"][ii]) for ii in where_val
        ],
    }
    test = {
        "grid": data["grid"],
        "x": data["x"][where_test, :],
        "y": data["y"][where_test, :],
        "names": [data["names"][ii] for ii in where_test],
        "origin_smiles": [data["origin_smiles"][ii] for ii in where_test],
        "n_atoms": [
            atom_count(data["origin_smiles"][ii]) for ii in where_test
        ],
    }

    L1 = len(train["origin_smiles"])
    L2 = len(val["origin_smiles"])
    L3 = len(test["origin_smiles"])
    print(f"Done with {L1} train, {L2} val and {L3} test")

    return {"train": train, "val": val, "test": test}


def random_split(data, prop_test=0.1, prop_val=0.1, seed=123):
    """Executes a fully random split of the provided data. The provided
    ``prop_test`` indicates the proportion of the data to reserve for testing.

    Parameters
    ----------
    data : dict
    prop_test : float, optional
    seed : int, optional

    Returns
    -------
    dict
    """

    L = len(data["x"])
    n_test = int(L * prop_test)
    n_val = int(L * prop_val)
    n_train = L - n_test - n_val
    assert n_train > 0
    _train, _val, _test = torch.utils.data.random_split(
        range(L),
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed) if seed is not None
        else None
    )
    where_train = _train.indices
    where_val = _val.indices
    where_test = _test.indices

    return _qm9_train_val_test_from_data(
        data, where_train, where_val, where_test
    )


def split_qm9_data_by_number_of_total_atoms(
    data,
    max_training_atoms_per_molecule=7,
    prop_val=0.1,
    seed=123
):
    """A helper function for preparing molecular data that resolves the
    training and testing sets by the number of total atoms in the molecule.

    .. note::

        It is assumed that the keys in the passed data are
        ``['grid', 'y', 'x', 'names', 'origin_smiles']``.

    Parameters
    ----------
    data : dict
        A dictionary with keys "``x``" and "``y``", at least, for the features
        and targets, respectively. More keys are required for additional
        functionality.
    max_training_atoms_per_molecule : int, optional
        Description
    prop_val : float, optional
        The proportion of the training set to use for cross-validation.
    seed : None, optional
        Deterministic split of the training and validation sets.

    Returns
    -------
    dict
        A dictionary of of dictionaries, with keys as "train" and "test".
    """

    print(
        "Parsing the qm9 data by number of total atoms="
        f"{max_training_atoms_per_molecule}"
    )

    n_absorbers_in_datapoint = np.array([
        sum(dict(atom_count(smi)).values()) for smi in data["origin_smiles"]
    ])

    _where_train = np.where(
        (n_absorbers_in_datapoint <= max_training_atoms_per_molecule)
    )[0]

    L = len(_where_train)
    n_val = int(L * prop_val)
    n_train = L - n_val
    assert n_train > 0
    _train, _val = torch.utils.data.random_split(
        range(L),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed) if seed is not None
        else None
    )
    where_train = _where_train[_train.indices]
    where_val = _where_train[_val.indices]
    where_test = np.where(
        (n_absorbers_in_datapoint > max_training_atoms_per_molecule)
    )[0]

    d = _qm9_train_val_test_from_data(
        data, where_train, where_val, where_test
    )

    # Triple check
    assert set(d["train"]["names"]).isdisjoint(set(d["val"]["names"]))
    assert set(d["train"]["names"]).isdisjoint(set(d["test"]["names"]))
    assert set(d["val"]["names"]).isdisjoint(set(d["test"]["names"]))

    # Molecules i.e. SMILES should also be disjoint between the TEST and TRAIN
    # sets, NOT the VAL and TRAIN sets.
    assert set(d["train"]["names"]).isdisjoint(set(d["test"]["names"]))
    assert set(d["val"]["names"]).isdisjoint(set(d["test"]["names"]))

    # Final assertion
    assert all([
        xx <= max_training_atoms_per_molecule
        for xx in sum(d["train"]["n_atoms"].values())
    ])
    assert all([
        xx <= max_training_atoms_per_molecule
        for xx in sum(d["val"]["n_atoms"].values())
    ])
    assert all([
        xx > max_training_atoms_per_molecule
        for xx in sum(d["test"]["n_atoms"].values())
    ])

    return d


def split_qm9_data_by_number_of_absorbers(
    data,
    absorber,
    max_training_absorbers=2,
    keep_zwitter=False
):
    """A helper function for preparing qm9 data that resolves the training and
    testing sets by the number of absorbing atoms. Specifically, the specified
    max number of absorbers is for the training set, and the rest of the
    data goes into the testing set.

    .. note::

        It is assumed that the keys in the passed data are
        ``['grid', 'y', 'x', 'names', 'origin_smiles']``.

    Parameters
    ----------
    data : dict
        A dictionary with keys "``x``" and "``y``", at least, for the features
        and targets, respectively. More keys are required for additional
        functionality.
    absorber : str
        The absorbing atom type.
    max_training_absorbers : int, optional
        The maximum number of absorbing atoms per molecule in the training set.
    keep_zwitter : bool, optional
        If False, will remove zwitter-ionic species.

    Returns
    -------
    dict
        A dictionary of of dictionaries, with keys as "train" and "test".
    """

    print(f"Parsing the qm9 data by number of absorbers={absorber}")
    print(f"Training data will have <={max_training_absorbers} absorbers")
    print("Test data will get the rest")
    print(f"Keeping zwitterions: {keep_zwitter}")

    # Will throw a key error if ``origin_smiles`` is not in the data
    unique_smiles = list(set(data["origin_smiles"]))

    # Convert to mol
    smiles_to_mol_map = {
        smile: Chem.MolFromSmiles(smile) for smile in unique_smiles
    }

    # Get the number of absorbers
    print("Computing smiles_to_n_absorbers_map")
    smiles_to_n_absorbers_map = dict()

    for smile in tqdm(smiles_to_mol_map.keys()):
        mol = smiles_to_mol_map[smile]
        counter = Counter([atom.GetSymbol() for atom in mol.GetAtoms()])
        smiles_to_n_absorbers_map[smile] = counter[absorber]

    n_absorbers_in_datapoint = np.array([
        smiles_to_n_absorbers_map[smi] for smi in data["origin_smiles"]
    ])

    where_train = np.where(
        (n_absorbers_in_datapoint <= max_training_absorbers)
    )[0]
    where_test = np.where(
        (n_absorbers_in_datapoint > max_training_absorbers)
    )[0]
    assert set(list(where_train)).isdisjoint(set(list(where_test)))

    # Now we split these up
    train = {
        "grid": data["grid"],
        "x": data["x"][where_train, :],
        "y": data["y"][where_train, :],
        "names": [data["names"][ii] for ii in where_train],
        "origin_smiles": [data["origin_smiles"][ii] for ii in where_train],
        "n_absorbers_in_molecule": [
            n_absorbers_in_datapoint[ii] for ii in where_train
        ],
    }
    test = {
        "grid": data["grid"],
        "x": data["x"][where_test, :],
        "y": data["y"][where_test, :],
        "names": [data["names"][ii] for ii in where_test],
        "origin_smiles": [data["origin_smiles"][ii] for ii in where_test],
        "n_absorbers_in_molecule": [
            n_absorbers_in_datapoint[ii] for ii in where_test
        ],
    }

    # Triple check
    assert set(train["names"]).isdisjoint(set(test["names"]))

    # Molecules i.e. SMILES should also be disjoint
    assert set(train["origin_smiles"]).isdisjoint(set(test["origin_smiles"]))

    # Final assertion
    assert all([
        xx <= max_training_absorbers for xx in train["n_absorbers_in_molecule"]
    ])
    assert all([
        xx > max_training_absorbers for xx in test["n_absorbers_in_molecule"]
    ])

    L1 = len(train["origin_smiles"])
    L2 = len(test["origin_smiles"])
    print(f"Done with {L1} train and {L2} test")

    return {"train": train, "test": test}
