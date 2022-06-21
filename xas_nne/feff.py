from pathlib import Path
from uuid import uuid4
import warnings

import numpy as np


ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
    "Uue": 119,
}


XANES_cards = """
EDGE      K
S02       1.0
COREHOLE  RPA
CONTROL   1 1 1 1 1 1

XANES     4 0.04 0.1

FMS       9.0
EXCHANGE  0 0.0 0.0 2
SCF       7.0 1 100 0.2 3
RPATH     -1
"""

EXAFS_cards = """
EDGE      K
S02       1.0
COREHOLE  RPA
CONTROL   1 1 1 1 1 1

EXAFS 20

EXCHANGE  0 0.0 0.0 0
SCF       7.0 1 100 0.2 3
RPATH     10
"""


class FeffWriter:
    @property
    def elements(self):
        return [str(xx.specie) for xx in self._molecule.sites]

    @property
    def coordinates(self):
        return np.array([xx.coords for xx in self._molecule.sites]).tolist()

    def __init__(self, molecule, xanes=True, name=None):
        self._molecule = molecule
        if xanes:
            self._cards = XANES_cards
        else:
            self._cards = EXAFS_cards
        self._name = name if name is not None else str(uuid4())

    def write_feff_inputs(self, target_directory, absorber):
        """Summary

        Parameters
        ----------
        target_directory : TYPE
            Description
        feff_config : TYPE
            Description
        name : TYPE
            Description

        Returns
        -------
        list of str
            List of all target directories written to. In other words, if
            the returned list, L, is given by e.g. L[ii] = "my/path", then
            there was a file "my/path/feff.inp" written.
        """

        elements = self.elements

        # Get the unique elements in this cluster
        unique_elements = list(set(elements))
        unique_elements = {e: ATOMIC_NUMBERS[e] for e in unique_elements}

        # Iterate through all of the absorbing sites. Each one will get
        # its own feff calculation
        absorbing_sites = [
            ii for ii, atom in enumerate(elements) if atom == absorber
        ]

        # Get the first lines corresponding to the potential
        N_abs = len(absorbing_sites)

        # Reference the potential lines in this dictionary
        potential_reference = dict()

        # Iterate through all absorbing sites in the xyz file
        for site in absorbing_sites:

            # Get the lines corresponding to the potential, starting with
            # the absorbing atom
            atomic_number = ATOMIC_NUMBERS[absorber]
            potentials_lines = [
                f"0\t{atomic_number}\t{absorber}\t-1\t-1\t{N_abs}"
            ]

            star_index = 1
            for element, atomic_number in unique_elements.items():

                # In some cases, we just have a single absorber and
                # therefore just one potential for it. Otherwise, we need
                # to specify that the other atoms of the same type, but
                # that are non-absorbing, have a different potential
                if element == absorber:
                    if len(absorbing_sites) == 1:
                        continue

                N_this_element = len(
                    [ii for ii, atom in enumerate(elements) if atom == element]
                )

                if N_this_element == 0:
                    continue

                atomic_number = ATOMIC_NUMBERS[element]
                potentials_lines.append(
                    f"{star_index}\t{atomic_number}\t{element}\t"
                    f"-1\t-1\t{N_this_element}"
                )

                # The potential_reference indexes everything except the
                # absorber which is always zero
                potential_reference[element] = star_index

                star_index += 1

            site_target_directory = target_directory / Path(
                f"{self._name}_{absorber}_{site:03}"
            )
            site_target_directory.mkdir(exist_ok=True, parents=True)
            path = site_target_directory / Path("feff.inp")

            with open(path, "w") as f:

                # First write the cards for the input files
                f.write(f"TITLE {self._name}_{absorber}_{site:03}\n")
                f.write(f"{self._cards}\n")
                f.write("POTENTIALS\n")
                f.write("*\tipot\tZ\telement\tl_scmt\tl_fms\n")

                # Write the potential lines
                for line in potentials_lines:
                    f.write(f"{line}\n")
                f.write("\n")

                # Write the coordinate lines and potential indexes
                f.write("ATOMS\n")
                zipped = zip(self.coordinates, elements)
                for ii, (c, e) in enumerate(zipped):
                    c_lines = f"{c[0]:.08f} {c[1]:.08f} {c[2]:.08f}"
                    if ii == site:
                        f.write(f"{c_lines} 0\n")
                        continue
                    n = potential_reference[e]
                    f.write(f"{c_lines} {n}\n")
                f.write("END\n")

                
def load_completed_FEFF_results(path):
    """Loads in the feff.in, feff.out and xmu.dat files. Parses the results
    into a dictionary.

    Parameters
    ----------
    path : os.PathLike
        The path to the directory containing ``feff.in``, ``feff.out`` and
        ``xmu.dat``.

    Returns
    -------
    dict
    """
    
    xmu_path = path / Path("xmu.dat")
    if not xmu_path.exists() or xmu_path.stat().st_size == 0:
        spectrum = None
        spectrum_metadata = None
    else:
        with warnings.catch_warnings(record=True):
            spectrum = np.loadtxt(xmu_path, comments="#").tolist()            
        if len(spectrum) == 0:
            spectrum = None
        with open(path / Path("xmu.dat")) as f:
            spectrum_metadata = f.readlines()
            spectrum_metadata = [
                xx for xx in spectrum_metadata if xx.startswith("#")
            ]

    with open(path / Path("feff.inp"), "r") as f:
        feff_inp = f.readlines()
    with open(path / Path("feff.out"), "r") as f:
        feff_out = f.readlines()
    
    info = path.parts[-1].split("_")
    qm9_id = str(int(info[0]))
    absorbing_atom = info[1]
    site = int(info[2])
    
    return {
        "spectrum": spectrum,
        "spectrum_metadata": spectrum_metadata,
        "feff.inp": feff_inp,
        "feff.out": feff_out,
        "qm9id": qm9_id,
        "absorbing_atom": absorbing_atom,
        "site": site
    }
