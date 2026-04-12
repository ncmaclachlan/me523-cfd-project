"""Load NS solver CSV output into 2D numpy arrays."""

import glob
import os

import numpy as np


def _load_field(path):
    """Read a single CSV file and reshape into 2D arrays.

    The C++ writer loops i (outer) then j (inner), so the flat data
    reshapes to (ni, nj) in C-order.  We transpose to (nj, ni) so that
    the first axis is y and the second is x, matching matplotlib's
    contourf convention.
    """
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    i_col = raw[:, 0].astype(int)
    j_col = raw[:, 1].astype(int)

    ni = i_col.max() + 1
    nj = j_col.max() + 1

    x = raw[:, 2].reshape(ni, nj).T
    y = raw[:, 3].reshape(ni, nj).T
    val = raw[:, 4].reshape(ni, nj).T

    return {"x": x, "y": y, "val": val, "ni": ni, "nj": nj}


def _find_field_file(directory, suffix):
    """Find a CSV file ending with the given suffix in directory."""
    matches = glob.glob(os.path.join(directory, f"*_{suffix}.csv"))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No *_{suffix}.csv file found in {directory}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple *_{suffix}.csv files found in {directory}: {matches}")
    return matches[0]


def load_snapshot(directory):
    """Load pressure, u-velocity, and v-velocity fields from a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing ``*_p.csv``, ``*_u.csv``,
        ``*_v.csv`` files.

    Returns
    -------
    dict
        Keys ``"p"``, ``"u"``, ``"v"`` each mapping to a dict with
        ``"x"``, ``"y"``, ``"val"`` (2D arrays) and ``"ni"``, ``"nj"``.
        Also includes ``"nx"`` and ``"ny"`` (pressure cell counts).
    """
    p = _load_field(_find_field_file(directory, "p"))
    u = _load_field(_find_field_file(directory, "u"))
    v = _load_field(_find_field_file(directory, "v"))

    return {"p": p, "u": u, "v": v, "nx": p["ni"], "ny": p["nj"]}
