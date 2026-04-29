"""Scaling study plots from run_stats.json profiling output."""

import json
import os

import numpy as np
import matplotlib.pyplot as plt


def load_run_stats(run_dir):
    """Load run_stats.json from a run directory.

    Parameters
    ----------
    run_dir : str
        Path to run directory (e.g. ``data/run_64_64_100_cfl0.5``).

    Returns
    -------
    dict
        Parsed JSON: keys ``"config"``, ``"timing"``, ``"solver_stats"``.

    Raises
    ------
    FileNotFoundError
        If ``run_stats.json`` is not present in ``run_dir``.
    """
    path = os.path.join(run_dir, "run_stats.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"run_stats.json not found in {run_dir}")
    with open(path) as fh:
        return json.load(fh)


def plot_scaling(run_dirs, labels, *, ax=None, ref_slope=True):
    """Log-log wall time vs N^2 scaling plot.

    Parameters
    ----------
    run_dirs : list of str
        Run directories in order of increasing grid size.
    labels : list of str
        Legend labels (same length as ``run_dirs``).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.
    ref_slope : bool
        Overlay an O(N^2) reference line (default True).

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    n2_vals = []
    wall_vals = []

    for run_dir, label in zip(run_dirs, labels):
        d = load_run_stats(run_dir)
        n2 = d["config"]["nx"] * d["config"]["ny"]
        wall = d["timing"]["wall_total_s"]
        n2_vals.append(n2)
        wall_vals.append(wall)
        ax.scatter(n2, wall, zorder=5, label=label)

    if ref_slope and len(n2_vals) >= 2:
        n2_arr = np.array(n2_vals, dtype=float)
        w_arr = np.array(wall_vals, dtype=float)
        c = w_arr[0] / n2_arr[0]
        n2_ref = np.array([n2_arr.min(), n2_arr.max()])
        ax.plot(n2_ref, c * n2_ref, color="gray", linestyle="--", label=r"$O(N^2)$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N^2 = n_x \times n_y$")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Scaling: Wall Time vs. Grid Size")
    ax.legend()

    return fig, ax


def plot_stage_breakdown(run_dir, *, ax=None):
    """Horizontal stacked bar showing solver-stage wall-time percentages.

    Parameters
    ----------
    run_dir : str
        Run directory containing ``run_stats.json``.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if None.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 1.8))
    else:
        fig = ax.figure

    d = load_run_stats(run_dir)
    t = d["timing"]
    total = t["wall_total_s"]

    stages = [
        ("BC",             t["wall_bc_s"]),
        ("Predict",        t["wall_predict_s"]),
        ("Pressure RHS",   t["wall_pressure_rhs_s"]),
        ("Pressure Solve", t["wall_pressure_solve_s"]),
        ("Correct",        t["wall_correct_s"]),
    ]

    pcts = [100.0 * w / total for _, w in stages]
    names = [n for n, _ in stages]
    colors = plt.cm.tab10.colors

    left = 0.0
    for name, pct, color in zip(names, pcts, colors):
        ax.barh(0, pct, left=left, color=color, label=name)
        if pct > 4.0:
            ax.text(left + pct / 2.0, 0, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=7, color="white")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of wall time (%)")
    ax.set_yticks([])
    nx = d["config"]["nx"]
    ny = d["config"]["ny"]
    ax.set_title(rf"Stage Breakdown: ${nx}\times{ny}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3),
              ncol=5, fontsize=7)

    return fig, ax


def plot_backend_comparison(run_dirs, labels, *, ax=None):
    """Bar chart comparing wall time per step across runs or backends.

    Parameters
    ----------
    run_dirs : list of str
        Run directories, each from a different backend or configuration.
    labels : list of str
        Bar labels.
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

    wall_per_step = []
    backends = []

    for run_dir in run_dirs:
        d = load_run_stats(run_dir)
        wall_per_step.append(d["timing"]["wall_per_step_ms"])
        backends.append(d["config"]["backend"])

    x = np.arange(len(labels))
    colors = list(plt.cm.tab10.colors)[:len(labels)]
    bars = ax.bar(x, wall_per_step, width=0.5, color=colors)
    ax.bar_label(bars, fmt="%.2f ms", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Wall time / step (ms)")
    ax.set_title("Backend / Configuration Comparison")

    return fig, ax
