"""CLI entry point: python -m plotting data/run_64_64_100_0.01 [--show]"""

import argparse
import os
import re

from . import (
    load_snapshot,
    plot_pressure,
    plot_velocity_magnitude,
    load_ke,
    load_divergence,
    load_error_norms,
    plot_kinetic_energy,
    plot_divergence,
    plot_error_norms,
)


def _parse_run_info(run_dir):
    """Extract grid size and Re from run directory name.

    Expected format: run_{nx}_{ny}_{Re}_{dt}
    Returns a title suffix string like "(64x64, Re=100)".
    """
    basename = os.path.basename(os.path.normpath(run_dir))
    m = re.match(r"run_(\d+)_(\d+)_(\d+)", basename)
    if m:
        nx, ny, re_num = m.group(1), m.group(2), m.group(3)
        return rf" (${nx}\times{ny}$, $Re={re_num}$)"
    return ""


def _append_title(ax, suffix):
    """Append suffix to an axes' existing title."""
    ax.set_title(ax.get_title() + suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Plot NS solver output fields")
    parser.add_argument("run_dir",
                        help="Run directory (e.g. data/run_64_64_100_0.01)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--format", default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output image format")
    args = parser.parse_args()

    output_dir = os.path.join(args.run_dir, "output")
    figures_dir = os.path.join(args.run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    suffix = _parse_run_info(args.run_dir)

    data = load_snapshot(output_dir)

    fig, ax, _ = plot_pressure(data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"pressure.{args.format}"))

    fig, ax, _ = plot_velocity_magnitude(data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"velocity_magnitude.{args.format}"))

    ke_data = load_ke(output_dir)
    div_data = load_divergence(output_dir)

    fig, ax = plot_kinetic_energy(ke_data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"kinetic_energy.{args.format}"))

    fig, ax = plot_divergence(div_data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"divergence.{args.format}"))

    error_csv = os.path.join(output_dir, "output_error.csv")
    if os.path.isfile(error_csv):
        err_data = load_error_norms(output_dir)
        fig, ax = plot_error_norms(err_data)
        _append_title(ax, suffix)
        fig.savefig(os.path.join(figures_dir, f"error_norms.{args.format}"))

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print(f"Saved plots to {figures_dir}/")


if __name__ == "__main__":
    main()
