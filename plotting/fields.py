"""Contour plot functions for NS solver fields."""

import numpy as np
import matplotlib.pyplot as plt

from .style import CMAP_PRESSURE, CMAP_VELOCITY, N_LEVELS


def _contour_field(x, y, z, *, cmap, title, label, levels=N_LEVELS,
                   ax=None, vmin=None, vmax=None):
    """Filled contour plot with colorbar.

    Returns (fig, ax, contour_set).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if isinstance(levels, int):
        lo = vmin if vmin is not None else z.min()
        hi = vmax if vmax is not None else z.max()
        if lo == hi:
            hi = lo + 1.0
        levels = np.linspace(lo, hi, levels)

    cs = ax.contourf(x, y, z, levels=levels, cmap=cmap)
    cb = fig.colorbar(cs, ax=ax, label=label, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, direction="in")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    ax.set_aspect("equal")

    return fig, ax, cs


def plot_pressure(data, *, ax=None, levels=N_LEVELS):
    """Filled contour of the pressure field.

    Uses a diverging colormap centered on zero.
    """
    p = data["p"]
    vabs = np.abs(p["val"]).max()
    if vabs == 0:
        vabs = 1.0

    return _contour_field(
        p["x"], p["y"], p["val"],
        cmap=CMAP_PRESSURE,
        title="Pressure",
        label=r"$p$",
        levels=levels,
        ax=ax,
        vmin=-vabs,
        vmax=vabs,
    )


def plot_velocity_magnitude(data, *, ax=None, levels=N_LEVELS):
    """Filled contour of velocity magnitude.

    Interpolates staggered u, v to cell centers before computing
    |V| = sqrt(u_cc^2 + v_cc^2).
    """
    u = data["u"]["val"]  # shape (ny, nx+1)
    v = data["v"]["val"]  # shape (ny+1, nx)

    u_cc = 0.5 * (u[:, :-1] + u[:, 1:])
    v_cc = 0.5 * (v[:-1, :] + v[1:, :])
    vmag = np.sqrt(u_cc**2 + v_cc**2)

    return _contour_field(
        data["p"]["x"], data["p"]["y"], vmag,
        cmap=CMAP_VELOCITY,
        title="Velocity Magnitude",
        label=r"$|\mathbf{V}|$",
        levels=levels,
        ax=ax,
        vmin=0,
    )
