"""CLI entry point: python -m plotting data/run_64_64_100_0.01 [--show]"""

import argparse
import os

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

    data = load_snapshot(output_dir)

    fig, _, _ = plot_pressure(data)
    fig.savefig(os.path.join(figures_dir, f"pressure.{args.format}"))

    fig, _, _ = plot_velocity_magnitude(data)
    fig.savefig(os.path.join(figures_dir, f"velocity_magnitude.{args.format}"))

    ke_data = load_ke(output_dir)
    div_data = load_divergence(output_dir)

    fig, _ = plot_kinetic_energy(ke_data)
    fig.savefig(os.path.join(figures_dir, f"kinetic_energy.{args.format}"))

    fig, _ = plot_divergence(div_data)
    fig.savefig(os.path.join(figures_dir, f"divergence.{args.format}"))

    error_csv = os.path.join(output_dir, "output_error.csv")
    if os.path.isfile(error_csv):
        err_data = load_error_norms(output_dir)
        fig, _ = plot_error_norms(err_data)
        fig.savefig(os.path.join(figures_dir, f"error_norms.{args.format}"))

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print(f"Saved plots to {figures_dir}/")


if __name__ == "__main__":
    main()
