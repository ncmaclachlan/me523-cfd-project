"""Plotting library for ME 523 NS solver output."""

from . import style  # noqa: F401  — applies rcParams on import
from .loader import load_snapshot
from .fields import plot_pressure, plot_velocity_magnitude, plot_velocity_error
from .diagnostics import (
    load_ke,
    load_divergence,
    load_error_norms,
    plot_kinetic_energy,
    plot_divergence,
    plot_error_norms,
    plot_diagnostics,
)

__all__ = [
    "load_snapshot",
    "plot_pressure",
    "plot_velocity_magnitude",
    "plot_velocity_error",
    "load_ke",
    "load_divergence",
    "load_error_norms",
    "plot_kinetic_energy",
    "plot_divergence",
    "plot_error_norms",
    "plot_diagnostics",
]
