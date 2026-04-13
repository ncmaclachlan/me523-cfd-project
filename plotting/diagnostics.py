"""Time-history plots for kinetic energy and divergence diagnostics."""

import os

import matplotlib.pyplot as plt
import numpy as np


def load_ke(directory):
    """Load kinetic energy time history from output_ke.csv.

    Parameters
    ----------
    directory : str
        Path to output directory containing ``output_ke.csv``.

    Returns
    -------
    dict
        Keys ``"step"``, ``"time"``, ``"kinetic_energy"`` as 1D arrays.
    """
    path = os.path.join(directory, "output_ke.csv")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return {
        "step": raw[:, 0].astype(int),
        "time": raw[:, 1],
        "kinetic_energy": raw[:, 2],
    }


def load_divergence(directory):
    """Load divergence L2-norm time history from output_div.csv.

    Parameters
    ----------
    directory : str
        Path to output directory containing ``output_div.csv``.

    Returns
    -------
    dict
        Keys ``"step"``, ``"time"``, ``"l2_divergence"`` as 1D arrays.
    """
    path = os.path.join(directory, "output_div.csv")
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    return {
        "step": raw[:, 0].astype(int),
        "time": raw[:, 1],
        "l2_divergence": raw[:, 2],
    }


def plot_kinetic_energy(ke_data, *, ax=None):
    """Plot kinetic energy vs time.

    Parameters
    ----------
    ke_data : dict
        Output of :func:`load_ke`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(ke_data["time"], ke_data["kinetic_energy"], linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic Energy")
    ax.set_title("Kinetic Energy Time History")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))

    return fig, ax


def plot_divergence(div_data, *, ax=None, log_scale=True):
    """Plot L2 divergence norm vs time.

    Parameters
    ----------
    div_data : dict
        Output of :func:`load_divergence`.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.
    log_scale : bool
        Use log scale on y-axis (default True, since divergence is
        typically near machine epsilon).

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(div_data["time"], div_data["l2_divergence"], linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\| \nabla \cdot \mathbf{u} \|_2$")
    ax.set_title("Divergence Time History")
    if log_scale:
        ax.set_yscale("log")

    return fig, ax


def plot_diagnostics(ke_data, div_data):
    """Two-panel figure with kinetic energy (top) and divergence (bottom).

    Parameters
    ----------
    ke_data : dict
        Output of :func:`load_ke`.
    div_data : dict
        Output of :func:`load_divergence`.

    Returns
    -------
    fig, (ax_ke, ax_div)
    """
    fig, (ax_ke, ax_div) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    plot_kinetic_energy(ke_data, ax=ax_ke)
    ax_ke.set_xlabel("")  # shared x-axis, label only on bottom

    plot_divergence(div_data, ax=ax_div)

    fig.tight_layout()
    return fig, (ax_ke, ax_div)
