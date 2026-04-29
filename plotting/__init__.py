"""Plotting library for ME 523 NS solver output."""

from . import style  # noqa: F401  — applies rcParams on import
from .loader import load_snapshot
from .fields import plot_pressure, plot_velocity_magnitude, plot_velocity_error
from .diagnostics import (
    load_ke,
    load_divergence,
    load_error_norms,
    load_timestep,
    plot_kinetic_energy,
    plot_divergence,
    plot_error_norms,
    plot_timestep,
    plot_diagnostics,
)
from .scaling import (
    load_run_stats,
    plot_scaling,
    plot_stage_breakdown,
    plot_backend_comparison,
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
    "load_timestep",
    "plot_error_norms",
    "plot_timestep",
    "plot_diagnostics",
    "load_run_stats",
    "plot_scaling",
    "plot_stage_breakdown",
    "plot_backend_comparison",
]
