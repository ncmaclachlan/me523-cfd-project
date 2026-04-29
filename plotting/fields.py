"""Contour plot functions for NS solver fields."""

import numpy as np
import matplotlib.pyplot as plt

from .style import CMAP_PRESSURE, CMAP_VELOCITY, N_LEVELS


def _contour_field(x, y, z, *, cmap, title, label, levels=N_LEVELS,
                   ax=None, vmin=None, vmax=None, n_ticks=7, tick_fmt=".1e"):
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
    ticks = np.linspace(levels[0], levels[-1], n_ticks)
    cb = fig.colorbar(cs, ax=ax, label=label, fraction=0.046, pad=0.04,
                      ticks=ticks)
    cb.ax.tick_params(labelsize=8, direction="in")
    cb.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: format(v, tick_fmt)))
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
        tick_fmt=".2f",
    )


def plot_velocity_error(data, t, re, *, levels=N_LEVELS):
    """Two-panel contour plot of u- and v-velocity errors vs. Taylor-Green exact.

    Computes the exact solution at the staggered face locations already stored
    in ``data["u"]`` and ``data["v"]``, then interpolates both numerical and
    exact fields to cell centres before differencing.

    Parameters
    ----------
    data : dict
        Output of :func:`~plotting.loader.load_snapshot`.
    t : float
        Simulation time at which the snapshot was taken.
    re : float
        Reynolds number (used for the exp(-2t/Re) decay factor).
    levels : int
        Number of contour levels.

    Returns
    -------
    fig, (ax_u, ax_v)
    """
    decay = np.exp(-2.0 * t / re)

    # ---- u error at u-face locations, then avg to cell centres ----
    xu = data["u"]["x"]          # shape (ny, nx+1)
    yu = data["u"]["y"]
    u_num = data["u"]["val"]
    u_exact_face = np.sin(xu) * np.cos(yu) * decay
    u_err_face   = u_num - u_exact_face
    u_err_cc     = 0.5 * (u_err_face[:, :-1] + u_err_face[:, 1:])

    # ---- v error at v-face locations, then avg to cell centres ----
    xv = data["v"]["x"]          # shape (ny+1, nx)
    yv = data["v"]["y"]
    v_num = data["v"]["val"]
    v_exact_face = -np.cos(xv) * np.sin(yv) * decay
    v_err_face   = v_num - v_exact_face
    v_err_cc     = 0.5 * (v_err_face[:-1, :] + v_err_face[1:, :])

    xp = data["p"]["x"]
    yp = data["p"]["y"]

    # Shared symmetric color range so both panels are directly comparable.
    # Ticks at -vabs, 0, +vabs keep 0 anchored in the centre of the bar.
    vabs = max(np.abs(u_err_cc).max(), np.abs(v_err_cc).max()) or 1.0
    shared_levels = np.linspace(-vabs, vabs, levels)
    cb_ticks = list(np.linspace(-vabs, vabs, 7))

    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax, err, title in [
        (ax_u, u_err_cc, r"$u$ error"),
        (ax_v, v_err_cc, r"$v$ error"),
    ]:
        cs = ax.contourf(xp, yp, err, levels=shared_levels, cmap=CMAP_PRESSURE)
        cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, ticks=cb_ticks)
        cb.ax.tick_params(labelsize=8, direction="in")
        cb.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.1e}"))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_title(title)
        ax.set_aspect("equal")

    return fig, (ax_u, ax_v)


def plot_streamlines(data, *, ax=None, density=1.4, color_by_speed=True,
                     title="Streamlines"):
    """Streamplot of the velocity field on cell-centred (u, v).

    Staggered u (ny, nx+1) and v (ny+1, nx) are averaged in their staggered
    direction to land on the pressure grid before being passed to
    matplotlib's streamplot, which requires 1D monotonically-increasing
    coordinate axes.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    u = data["u"]["val"]
    v = data["v"]["val"]
    u_cc = 0.5 * (u[:, :-1] + u[:, 1:])
    v_cc = 0.5 * (v[:-1, :] + v[1:, :])

    # 1D coordinate axes from the pressure grid (which is cell-centred).
    x_cc = data["p"]["x"][0, :]
    y_cc = data["p"]["y"][:, 0]

    if color_by_speed:
        speed = np.sqrt(u_cc**2 + v_cc**2)
        sp = ax.streamplot(x_cc, y_cc, u_cc, v_cc, color=speed,
                           cmap=CMAP_VELOCITY, density=density,
                           linewidth=0.6, arrowsize=0.7)
        fig.colorbar(sp.lines, ax=ax, label=r"$|\mathbf{V}|$",
                     fraction=0.046, pad=0.04)
    else:
        ax.streamplot(x_cc, y_cc, u_cc, v_cc, color="black",
                      density=density, linewidth=0.6, arrowsize=0.7)

    ax.set_xlim(x_cc[0], x_cc[-1])
    ax.set_ylim(y_cc[0], y_cc[-1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.set_title(title)
    return fig, ax


def plot_centerline_u(data, *, x_cut=None, ax=None, label=None,
                      title=r"$u(y)$ profile"):
    """1D profile of u along y at a fixed streamwise station x_cut.

    If x_cut is None the midline x = (x_min + x_max) / 2 is used.
    Picks the u-face column whose x is closest to x_cut so no
    interpolation is performed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 4.0))
    else:
        fig = ax.figure

    xu = data["u"]["x"][0, :]   # 1D x at u-faces
    yu = data["u"]["y"][:, 0]   # 1D y at u-faces

    if x_cut is None:
        x_cut = 0.5 * (xu[0] + xu[-1])

    j_cut = int(np.argmin(np.abs(xu - x_cut)))
    u_col = data["u"]["val"][:, j_cut]

    ax.plot(u_col, yu, label=label or rf"$x = {xu[j_cut]:.3f}$")
    ax.axvline(0.0, color="0.7", linewidth=0.5, zorder=0)
    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")
    ax.set_ylim(yu[0], yu[-1])
    ax.set_title(title)
    if label:
        ax.legend()
    return fig, ax


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
        vmax=1,
        tick_fmt=".2f",
    )
