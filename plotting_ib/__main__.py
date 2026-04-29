"""CLI: python -m plotting_ib data/run_128_128_30_cfl0.5 [--D 1.0 ...]

Generates a fixed set of figures into <run_dir>/figures_ib/:
    velocity_magnitude.png   |u| with cylinder mask
    pressure.png             p (zero-centred)
    vorticity.png            omega_z (zero-centred)
    streamlines.png          streamplot coloured by speed
    centerline_u.png         u(y) at x = Lx/2 and at x = xc + 2D
    kinetic_energy.png       KE(t)
    divergence.png           ||div u||_2(t) on log scale
    dke_dt.png               |dKE/dt|(t) -- steady-state diagnostic
"""

import argparse
import os
import re

from . import (
    load_snapshot,
    load_ke,
    load_divergence,
    plot_velocity_magnitude,
    plot_pressure,
    plot_vorticity,
    plot_streamlines,
    plot_centerline_u,
    plot_kinetic_energy,
    plot_divergence,
    plot_dkedt,
)


def _parse_run_info(run_dir):
    """Extract (Re, nx, cfl) from directory name `run_{nx}_{ny}_{Re}_cfl{cfl}`."""
    base = os.path.basename(os.path.normpath(run_dir))
    m = re.match(r"run_(\d+)_(\d+)_([\d.]+)_cfl([\d.]+)", base)
    if not m:
        return None, None, None
    return float(m.group(3)), int(m.group(1)), float(m.group(4))


def main():
    p = argparse.ArgumentParser(description="IB-solver plotting suite")
    p.add_argument("run_dir", help="data/run_..._cfl... directory")
    p.add_argument("--D",  type=float, default=1.0, help="Cylinder diameter")
    p.add_argument("--xc", type=float, default=None,
                   help="Cylinder centre x (default 2.5*D)")
    p.add_argument("--yc", type=float, default=None,
                   help="Cylinder centre y (default Ly/2)")
    p.add_argument("--no-cylinder", action="store_true",
                   help="Skip cylinder overlay (e.g. plain inflow/outflow runs)")
    p.add_argument("--format", default="png",
                   choices=["png", "pdf", "svg"])
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    output_dir = os.path.join(args.run_dir, "output")
    fig_dir    = os.path.join(args.run_dir, "figures_ib")
    os.makedirs(fig_dir, exist_ok=True)

    re_val, nx, cfl = _parse_run_info(args.run_dir)
    title_suffix = ""
    if re_val is not None:
        title_suffix = f"  ($Re={re_val:g}$, ${nx}^2$, CFL={cfl})"

    snap = load_snapshot(output_dir)
    ke   = load_ke(output_dir)
    div  = load_divergence(output_dir)

    R  = 0.5 * args.D
    xc = args.xc if args.xc is not None else 2.5 * args.D
    yc = args.yc if args.yc is not None else 0.5 * snap["Ly"]
    cyl = None if args.no_cylinder else (xc, yc, R)

    def save(fig, name):
        path = os.path.join(fig_dir, f"{name}.{args.format}")
        fig.savefig(path)
        print(f"  wrote {path}")

    # --- 2D fields ---
    fig, ax, _ = plot_velocity_magnitude(
        snap, cyl=cyl,
        title=r"Velocity magnitude $|\mathbf{u}|$" + title_suffix)
    save(fig, "velocity_magnitude")

    fig, ax, _ = plot_pressure(
        snap, cyl=cyl, title="Pressure $p$" + title_suffix)
    save(fig, "pressure")

    if snap["w"] is not None:
        fig, ax, _ = plot_vorticity(
            snap, cyl=cyl, title=r"Vorticity $\omega_z$" + title_suffix)
        save(fig, "vorticity")

    fig, ax = plot_streamlines(
        snap, cyl=cyl, title="Streamlines" + title_suffix)
    save(fig, "streamlines")

    # --- 1D centerline cut: project deliverable 1(g) ---
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    plot_centerline_u(snap, x_cut=0.5 * snap["Lx"], ax=ax,
                      label=fr"$x = L_x/2$",
                      title=r"$u(y)$ profile" + title_suffix)
    plot_centerline_u(snap, x_cut=xc + 2.0 * args.D, ax=ax,
                      label=fr"$x = x_c + 2D$")
    plot_centerline_u(snap, x_cut=xc - 1.0 * args.D, ax=ax,
                      label=fr"$x = x_c - D$")
    ax.legend(fontsize=8)
    save(fig, "centerline_u")

    # --- time histories ---
    fig, ax = plot_kinetic_energy(ke, title="Kinetic energy" + title_suffix)
    save(fig, "kinetic_energy")

    fig, ax = plot_divergence(div,
        title=r"$\|\nabla\cdot\mathbf{u}\|_2$" + title_suffix)
    save(fig, "divergence")

    fig, ax = plot_dkedt(ke,
        title=r"Steady-state proxy $|\mathrm{d}\mathrm{KE}/\mathrm{d}t|$"
              + title_suffix)
    save(fig, "dke_dt")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    print(f"\nAll plots written to {fig_dir}/")


if __name__ == "__main__":
    main()
