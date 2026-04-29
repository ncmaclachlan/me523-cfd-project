"""1D cuts through the velocity field (project deliverable 1(g))."""

import numpy as np
import matplotlib.pyplot as plt


def plot_centerline_u(snap, *, x_cut=None, ax=None, label=None,
                      title=None):
    """u(y) at a fixed streamwise station x_cut.

    Uses the u-face data directly (no interpolation in y) by selecting
    the column whose face x is closest to x_cut.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 4.0))
    else:
        fig = ax.figure

    if x_cut is None:
        x_cut = 0.5 * snap["Lx"]

    xu = snap["u"]["x"][0, :]      # 1D x at u-faces
    j_cut = int(np.argmin(np.abs(xu - x_cut)))
    u_col = snap["u"]["val"][:, j_cut]
    y_col = snap["u"]["y"][:, j_cut]

    ax.plot(u_col, y_col, label=label or rf"$x = {xu[j_cut]:.2f}$")
    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")
    ax.axvline(0.0, color="0.7", linewidth=0.5, zorder=0)
    ax.set_ylim(snap["y_cc"][0], snap["y_cc"][-1])
    if title:
        ax.set_title(title)
    if label:
        ax.legend()
    return fig, ax
