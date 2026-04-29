"""2D field plots for the IB solver.

All plots overlay the cylinder outline and respect the actual physical
domain extent (no Lx=2pi assumptions).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .style import CMAP_DIVERGING, CMAP_SEQUENTIAL, N_LEVELS


def _add_cylinder(ax, xc, yc, R, *, fill=True):
    if xc is None or yc is None or R is None:
        return
    # Solid white disc to mask whatever is inside (the IB residual is
    # not physically meaningful), with a thin black outline.
    if fill:
        ax.add_patch(Circle((xc, yc), R, facecolor="white",
                            edgecolor="black", linewidth=0.8, zorder=5))
    else:
        ax.add_patch(Circle((xc, yc), R, facecolor="none",
                            edgecolor="black", linewidth=0.8, zorder=5))


def _set_extent(ax, snap):
    x0 = snap["x_cc"][0]  - 0.5 * (snap["x_cc"][1] - snap["x_cc"][0])
    x1 = snap["x_cc"][-1] + 0.5 * (snap["x_cc"][1] - snap["x_cc"][0])
    y0 = snap["y_cc"][0]  - 0.5 * (snap["y_cc"][1] - snap["y_cc"][0])
    y1 = snap["y_cc"][-1] + 0.5 * (snap["y_cc"][1] - snap["y_cc"][0])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")


def plot_velocity_magnitude(snap, *, cyl=None, vmax=None, ax=None,
                            title="Velocity magnitude"):
    """|V| = sqrt(u_cc^2 + v_cc^2) at cell centres."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    vmag = np.hypot(snap["u_cc"], snap["v_cc"])
    if vmax is None:
        vmax = float(np.percentile(vmag, 99.5))
    levels = np.linspace(0.0, vmax, N_LEVELS)

    cs = ax.contourf(snap["p"]["x"], snap["p"]["y"], vmag,
                     levels=levels, cmap=CMAP_SEQUENTIAL, extend="max")
    fig.colorbar(cs, ax=ax, label=r"$|\mathbf{u}|$",
                 fraction=0.025, pad=0.02)
    _set_extent(ax, snap)
    if cyl is not None:
        _add_cylinder(ax, *cyl)
    ax.set_title(title)
    return fig, ax, cs


def plot_pressure(snap, *, cyl=None, vabs=None, ax=None,
                  title="Pressure"):
    """Diverging contour of p centred at zero."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    p = snap["p"]["val"]
    if vabs is None:
        vabs = float(np.percentile(np.abs(p), 99.0)) or 1.0
    levels = np.linspace(-vabs, vabs, N_LEVELS)

    cs = ax.contourf(snap["p"]["x"], snap["p"]["y"], p,
                     levels=levels, cmap=CMAP_DIVERGING, extend="both")
    fig.colorbar(cs, ax=ax, label=r"$p$", fraction=0.025, pad=0.02)
    _set_extent(ax, snap)
    if cyl is not None:
        _add_cylinder(ax, *cyl)
    ax.set_title(title)
    return fig, ax, cs


def plot_vorticity(snap, *, cyl=None, vabs=None, ax=None,
                   title="Vorticity"):
    """Diverging contour of omega = dv/dx - du/dy.

    Uses the corner-grid CSV from CSVOutput::write_vorticity.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if snap["w"] is None:
        raise FileNotFoundError("No vorticity CSV in this run -- "
                                "rerun ib_solver to generate output_w.csv")

    w = snap["w"]
    val = w["val"]
    if vabs is None:
        vabs = float(np.percentile(np.abs(val), 99.0)) or 1.0
    levels = np.linspace(-vabs, vabs, N_LEVELS)

    cs = ax.contourf(w["x"], w["y"], val,
                     levels=levels, cmap=CMAP_DIVERGING, extend="both")
    fig.colorbar(cs, ax=ax, label=r"$\omega_z$", fraction=0.025, pad=0.02)
    _set_extent(ax, snap)
    if cyl is not None:
        _add_cylinder(ax, *cyl, fill=False)
    ax.set_title(title)
    return fig, ax, cs


def plot_streamlines(snap, *, cyl=None, density=1.4, ax=None,
                     color_by_speed=True, title="Streamlines"):
    """matplotlib streamplot on cell-centred velocity."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    speed = np.hypot(snap["u_cc"], snap["v_cc"])
    kwargs = dict(density=density, linewidth=0.6, arrowsize=0.7)
    if color_by_speed:
        sp = ax.streamplot(snap["x_cc"], snap["y_cc"],
                           snap["u_cc"], snap["v_cc"],
                           color=speed, cmap=CMAP_SEQUENTIAL, **kwargs)
        fig.colorbar(sp.lines, ax=ax, label=r"$|\mathbf{u}|$",
                     fraction=0.025, pad=0.02)
    else:
        ax.streamplot(snap["x_cc"], snap["y_cc"],
                      snap["u_cc"], snap["v_cc"],
                      color="black", **kwargs)
    _set_extent(ax, snap)
    if cyl is not None:
        _add_cylinder(ax, *cyl)
    ax.set_title(title)
    return fig, ax
