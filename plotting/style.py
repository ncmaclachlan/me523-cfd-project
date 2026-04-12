"""Consistent matplotlib style for ME 523 CFD project plots."""

import matplotlib.pyplot as plt

STYLE = {
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.figsize": (6, 5),
    "figure.dpi": 150,
    "image.cmap": "viridis",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}
plt.rcParams.update(STYLE)

# Colormaps for specific plot types
CMAP_PRESSURE = "RdBu_r"
CMAP_VELOCITY = "viridis"

# Default number of contour levels
N_LEVELS = 32
