"""Consistent matplotlib style for ME 523 CFD project plots.

Journal-quality defaults: serif fonts (matching LaTeX), inward ticks on
all spines, high-resolution output suitable for publication.
"""

import matplotlib.pyplot as plt

STYLE = {
    # --- Fonts (match LaTeX documents) ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # --- Figure ---
    "figure.figsize": (3.5, 3.0),       # single-column journal width
    "figure.dpi": 300,
    "figure.constrained_layout.use": True,

    # --- Axes ---
    "axes.linewidth": 0.8,
    "axes.labelpad": 4,
    "axes.titlepad": 6,

    # --- Ticks (inward, all four spines) ---
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,

    # --- Lines ---
    "lines.linewidth": 1.0,
    "lines.markersize": 4,

    # --- Legend ---
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,

    # --- Grid (off by default, enable per-plot if needed) ---
    "axes.grid": False,

    # --- Images / colormaps ---
    "image.cmap": "viridis",

    # --- Saving ---
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}
plt.rcParams.update(STYLE)

# Colormaps for specific plot types
CMAP_PRESSURE = "RdBu_r"
CMAP_VELOCITY = "viridis"

# Default number of contour levels
N_LEVELS = 32
