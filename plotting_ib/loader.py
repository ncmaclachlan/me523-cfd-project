"""Load IB-solver CSV output into structured numpy arrays.

The C++ writer loops i (outer) then j (inner) for every CSV, so the flat
data reshapes to (ni, nj) in C-order. We transpose to (nj, ni) so axis-0
is y and axis-1 is x -- matching what matplotlib's contour/imshow want.
"""

import glob
import os

import numpy as np


def _load_field(path):
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    i = raw[:, 0].astype(int)
    j = raw[:, 1].astype(int)
    ni = i.max() + 1
    nj = j.max() + 1
    x   = raw[:, 2].reshape(ni, nj).T
    y   = raw[:, 3].reshape(ni, nj).T
    val = raw[:, 4].reshape(ni, nj).T
    return {"x": x, "y": y, "val": val, "ni": ni, "nj": nj}


def _find(directory, suffix):
    matches = glob.glob(os.path.join(directory, f"*_{suffix}.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_{suffix}.csv in {directory}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous *_{suffix}.csv in {directory}: {matches}")
    return matches[0]


def load_snapshot(directory):
    """Load p, u, v, and (if present) vorticity into one dict.

    Also computes cell-centred u_cc, v_cc and 1D coordinate arrays
    (x_cc, y_cc) used by streamlines, magnitude, and centerline cuts.
    """
    p = _load_field(_find(directory, "p"))
    u = _load_field(_find(directory, "u"))
    v = _load_field(_find(directory, "v"))

    # Interpolate staggered velocities to cell centres.
    # u has shape (ny, nx+1) -> avg adjacent columns -> (ny, nx)
    # v has shape (ny+1, nx) -> avg adjacent rows    -> (ny, nx)
    u_cc = 0.5 * (u["val"][:, :-1] + u["val"][:, 1:])
    v_cc = 0.5 * (v["val"][:-1, :] + v["val"][1:, :])

    # 1D cell-centre coordinate axes (matplotlib streamplot wants 1D
    # monotonically increasing arrays for x and y).
    x_cc = p["x"][0, :].copy()
    y_cc = p["y"][:, 0].copy()

    snap = {
        "p":    p,
        "u":    u,
        "v":    v,
        "u_cc": u_cc,
        "v_cc": v_cc,
        "x_cc": x_cc,
        "y_cc": y_cc,
        "nx":   p["ni"],
        "ny":   p["nj"],
        "Lx":   x_cc[-1] - x_cc[0] + (x_cc[1] - x_cc[0]),
        "Ly":   y_cc[-1] - y_cc[0] + (y_cc[1] - y_cc[0]),
    }

    # Optional vorticity (only present from the IB solver).
    try:
        snap["w"] = _load_field(_find(directory, "w"))
    except FileNotFoundError:
        snap["w"] = None

    return snap


def load_vorticity(directory):
    return _load_field(_find(directory, "w"))


def load_ke(directory):
    raw = np.loadtxt(_find(directory, "ke"), delimiter=",", skiprows=1)
    return {"step": raw[:, 0].astype(int),
            "time": raw[:, 1],
            "ke":   raw[:, 2]}


def load_divergence(directory):
    raw = np.loadtxt(_find(directory, "div"), delimiter=",", skiprows=1)
    return {"step": raw[:, 0].astype(int),
            "time": raw[:, 1],
            "div":  raw[:, 2]}
