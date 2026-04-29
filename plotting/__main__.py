"""CLI entry point: python -m plotting data/run_64_64_100_cfl0.49 [--show]"""

import argparse
import os
import re

from . import (
    load_snapshot,
    plot_pressure,
    plot_velocity_magnitude,
    plot_velocity_error,
    plot_streamlines,
    plot_centerline_u,
    load_ke,
    load_divergence,
    load_error_norms,
    load_timestep,
    plot_kinetic_energy,
    plot_divergence,
    plot_error_norms,
    plot_timestep,
)


def _parse_run_info(run_dir):
    """Extract grid size, Re, and CFL/dt from run directory name.

    Expected format: run_{nx}_{ny}_{Re}_cfl{cfl}  (or _dt{dt} for fixed-dt runs)
    Returns (title_suffix, re_value, nx_value, cfl_value).
    """
    basename = os.path.basename(os.path.normpath(run_dir))
    m_cfl = re.match(r"run_(\d+)_(\d+)_([\d.]+)_cfl([\d.]+)", basename)
    m_dt  = re.match(r"run_(\d+)_(\d+)_([\d.]+)_dt([\d.]+)",  basename)
    if m_cfl:
        nx, ny, re_num, cfl = m_cfl.group(1), m_cfl.group(2), m_cfl.group(3), m_cfl.group(4)
        suffix = rf" (${nx}\times{ny}$, $Re={re_num}$, $\mathrm{{CFL}}={cfl}$)"
        return suffix, float(re_num), int(nx), float(cfl)
    if m_dt:
        nx, ny, re_num, dt = m_dt.group(1), m_dt.group(2), m_dt.group(3), m_dt.group(4)
        suffix = rf" (${nx}\times{ny}$, $Re={re_num}$, $\Delta t={dt}$)"
        return suffix, float(re_num), int(nx), None
    # Fallback: old format run_{nx}_{ny}_{Re}_{dt} (no cfl/dt label)
    m = re.match(r"run_(\d+)_(\d+)_([\d.]+)", basename)
    if m:
        nx, ny, re_num = m.group(1), m.group(2), m.group(3)
        suffix = rf" (${nx}\times{ny}$, $Re={re_num}$)"
        return suffix, float(re_num), int(nx), None
    return "", None, None, None


def _append_title(ax, suffix):
    """Append suffix to an axes' existing title."""
    ax.set_title(ax.get_title() + suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Plot NS solver output fields")
    parser.add_argument("run_dir",
                        help="Run directory (e.g. data/run_64_64_100_cfl0.49)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    parser.add_argument("--format", default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output image format")
    parser.add_argument("--scaling", nargs="+", metavar="RUN_DIR",
                        help="Additional run dirs for scaling/comparison plots")
    args = parser.parse_args()

    output_dir = os.path.join(args.run_dir, "output")
    figures_dir = os.path.join(args.run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    suffix, re_val, nx_val, cfl_val = _parse_run_info(args.run_dir)

    data = load_snapshot(output_dir)

    fig, ax, _ = plot_pressure(data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"pressure.{args.format}"))

    fig, ax, _ = plot_velocity_magnitude(data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"velocity_magnitude.{args.format}"))

    fig, ax = plot_streamlines(data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"streamlines.{args.format}"))

    # For Taylor-Green, sin(x)=0 at x=0, pi, 2pi, so the midline gives
    # only roundoff. Cut at x = Lx/4 (= pi/2 for Lx = 2*pi) where the
    # profile has its maximum amplitude.
    x_left  = data["u"]["x"][0, 0]
    x_right = data["u"]["x"][0, -1]
    xu_cut  = x_left + 0.25 * (x_right - x_left)
    fig, ax = plot_centerline_u(data, x_cut=xu_cut)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"centerline_u.{args.format}"))

    ke_data = load_ke(output_dir)
    div_data = load_divergence(output_dir)

    if re_val is not None:
        t_end = ke_data["time"][-1]
        fig, (ax_u, ax_v) = plot_velocity_error(data, t_end, re_val)
        fig.suptitle("Velocity Error at $t = t_{{\\mathrm{{end}}}}$" + suffix,
                     fontsize=9)
        fig.savefig(os.path.join(figures_dir, f"velocity_error.{args.format}"))

    fig, ax = plot_kinetic_energy(ke_data, re=re_val)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"kinetic_energy.{args.format}"))

    fig, ax = plot_divergence(div_data)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"divergence.{args.format}"))

    import math
    ts_data = load_timestep(output_dir)
    lx = 2.0 * math.pi  # domain length
    fig, ax = plot_timestep(ts_data, cfl=cfl_val, nx=nx_val, lx=lx)
    _append_title(ax, suffix)
    fig.savefig(os.path.join(figures_dir, f"timestep.{args.format}"))

    error_csv = os.path.join(output_dir, "output_error.csv")
    if os.path.isfile(error_csv):
        err_data = load_error_norms(output_dir)
        fig, ax = plot_error_norms(err_data)
        _append_title(ax, suffix)
        fig.savefig(os.path.join(figures_dir, f"error_norms.{args.format}"))

    if args.scaling:
        from . import plot_scaling, plot_stage_breakdown, plot_backend_comparison
        all_dirs = [args.run_dir] + args.scaling

        def _grid_label(d):
            m = re.match(r"run_(\d+)_(\d+)", os.path.basename(os.path.normpath(d)))
            return f"{m.group(1)}×{m.group(2)}" if m else os.path.basename(os.path.normpath(d))

        labels = [_grid_label(d) for d in all_dirs]

        fig, ax = plot_scaling(all_dirs, labels)
        fig.savefig(os.path.join(figures_dir, f"scaling.{args.format}"),
                    bbox_inches="tight")

        fig, ax = plot_stage_breakdown(args.run_dir)
        fig.savefig(os.path.join(figures_dir, f"stage_breakdown.{args.format}"),
                    bbox_inches="tight")

        fig, ax = plot_backend_comparison(all_dirs, labels)
        fig.savefig(os.path.join(figures_dir, f"backend_comparison.{args.format}"),
                    bbox_inches="tight")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print(f"Saved plots to {figures_dir}/")


if __name__ == "__main__":
    main()
