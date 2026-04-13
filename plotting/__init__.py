"""Plotting library for ME 523 NS solver output."""

from . import style  # noqa: F401  — applies rcParams on import
from .loader import load_snapshot
from .fields import plot_pressure, plot_velocity_magnitude

__all__ = ["load_snapshot", "plot_pressure", "plot_velocity_magnitude"]
