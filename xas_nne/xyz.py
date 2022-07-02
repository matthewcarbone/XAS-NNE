import numpy as np


def process_extended_xyz_file_to_array(extended_xyz_file_path, verbose=True):
    """Processes an arbitrary extended xyz file to a numpy array.

    Parameters
    ----------
    extended_xyz_file_path : TYPE
        Description
    verbose : bool, optional
        Description

    Returns
    -------
    numpy.ndarray
        An array of 3 dimensions. The first dimension is the "time" or
        "snapshot" index. The second is the atom index, and the third is the
        spatial coordinate.
    """

    with open(extended_xyz_file_path, "r") as input_file:

        # Read all the lines at once
        lines = input_file.readlines()

        # Get the number of atoms per block, which is always the first line of
        # either an xyz or extended xyz file
        n_atoms = int(lines[0].strip())

    # We can print some diagnostics to help us debug
    if verbose:
        print(
            f"Read {len(lines)} lines from {extended_xyz_file_path}, each "
            f"block has {n_atoms} atoms"
        )

    # Each "single" xyz file has the following lines:
    # A single line indicating how many atoms there are in the block
    # A comment line
    # n_atoms lines for the species type and coordinates
    # With this information, we can "chunk" the list into some number of equal
    # parts each containing 12+2 lines.
    # Check out a way to do this here:
    # https://www.delftstack.com/howto/python/
    # python-split-list-into-chunks/
    # #split-list-in-python-to-chunks-using-the-lambda-function
    EXTRA_LINES = 2  # <- no magic numbers
    offset = n_atoms + EXTRA_LINES

    # List comprehension is much faster than for loops. Try to avoid the latter
    # when at all possible
    chunked = [lines[ii:ii + offset] for ii in range(0, len(lines), offset)]

    if verbose:
        print(f"Got {len(chunked)} snapshots")

    # Each entry of chunked contains the:
    # - number of atoms (same for everything)
    # - the energy (I think)
    # - the atom types/coordinates
    # Get the energies
    comment_lines = np.array([
        float(lines[ii + 1]) for ii in range(0, len(lines), offset)
    ])

    # Get the atom list - only have to do this once!
    atom_list = [line.split()[0] for line in chunked[0][EXTRA_LINES:]]

    # Finally, get the coordinates
    chunked = np.array([
        [line.split()[1:4] for line in chunk[EXTRA_LINES:]]
        for chunk in chunked
    ], dtype=float)

    return dict(energy=comment_lines, elements=atom_list, coordinates=chunked)
