"""Time-history plots: KE, divergence, steady-state detection."""

import numpy as np
import matplotlib.pyplot as plt


def plot_kinetic_energy(ke, *, ax=None, title="Kinetic energy"):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(ke["time"], ke["ke"])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{KE}(t)$")
    ax.set_title(title)
    return fig, ax


def plot_divergence(div, *, ax=None, title=r"$\|\nabla\cdot\mathbf{u}\|_2$"):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.semilogy(div["time"], np.maximum(div["div"], 1e-20))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|\nabla\cdot\mathbf{u}\|_2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax


def plot_dkedt(ke, *, ax=None, title="Steady-state proxy: |dKE/dt|"):
    """Numerical |dKE/dt| -- a plateau near zero indicates steady state."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    t = ke["time"]
    e = ke["ke"]
    if len(t) < 3:
        return fig, ax
    dedt = np.gradient(e, t)
    ax.semilogy(t, np.maximum(np.abs(dedt), 1e-20))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\mathrm{d}\,\mathrm{KE}/\mathrm{d}t|$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    return fig, ax
