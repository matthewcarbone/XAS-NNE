from contextlib import contextmanager
import json
import time


def read_json(path):
    with open(path, 'r') as infile:
        dat = json.load(infile)
    return dat


def save_json(d, path, indent=4, sort_keys=True):
    """Saves a json file to the path specified.

    Parameters
    ----------
    d : dict
        Must be serializable.
    path : str
        File path to save at.
    """

    with open(path, 'w') as outfile:
        json.dump(d, outfile, indent=indent, sort_keys=sort_keys)


@contextmanager
def timeit(msg):
    """A simple utility for timing how long a certain block of code will take.
    Results are printed in minutes.

    Parameters
    ----------
    msg : str
        The message to log.
    """

    t0 = time.time()
    try:
        yield None
    finally:
        dt = time.time() - t0
        dt = dt / 60.0
        print(f"[{dt:.01f} m] {msg}")
