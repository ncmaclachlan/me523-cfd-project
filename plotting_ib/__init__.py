"""Plotting suite for the immersed-boundary solver (apps/ib_solver).

Distinct from the periodic / Taylor-Green oriented `plotting/` package:
no exact-solution comparison, no Lx=2pi assumptions, native vorticity CSV
support, cylinder geometry overlays, streamlines, and centerline u(y) cuts
that match the deliverables in Final_report_instructions.pdf section 2.4.
"""

from . import style  # applies rcParams on import
from .loader  import load_snapshot, load_vorticity, load_ke, load_divergence
from .fields  import (
    plot_velocity_magnitude,
    plot_pressure,
    plot_vorticity,
    plot_streamlines,
)
from .centerline import plot_centerline_u
from .diagnostics import plot_kinetic_energy, plot_divergence, plot_dkedt

__all__ = [
    "load_snapshot",
    "load_vorticity",
    "load_ke",
    "load_divergence",
    "plot_velocity_magnitude",
    "plot_pressure",
    "plot_vorticity",
    "plot_streamlines",
    "plot_centerline_u",
    "plot_kinetic_energy",
    "plot_divergence",
    "plot_dkedt",
]
