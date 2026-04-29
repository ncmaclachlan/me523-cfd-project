"""Matplotlib style for the IB plotting suite.

Wider default figures because the cylinder domain is 10D x 10D --
square contour plots at single-column width make features illegible.
"""

import matplotlib.pyplot as plt

STYLE = {
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":      "cm",
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       9,

    "figure.figsize":        (6.0, 3.5),
    "figure.dpi":            150,
    "figure.constrained_layout.use": True,

    "axes.linewidth":        0.8,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.top":             True,
    "ytick.right":           True,
    "xtick.minor.visible":   True,
    "ytick.minor.visible":   True,

    "lines.linewidth":       1.0,
    "image.cmap":            "viridis",

    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
}
plt.rcParams.update(STYLE)

CMAP_DIVERGING = "RdBu_r"   # pressure, vorticity (zero-centred)
CMAP_SEQUENTIAL = "viridis" # velocity magnitude
N_LEVELS        = 41
