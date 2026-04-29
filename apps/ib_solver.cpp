#include <Kokkos_Core.hpp>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include "solver.hpp"
#include "immersed_boundary.hpp"

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --nx N            Grid cells in x (default 128)\n"
              << "  --ny N            Grid cells in y (default 128)\n"
              << "  --re F            Reynolds number  (default 200, based on U*D)\n"
              << "  --u_inf F         Inflow streamwise velocity (default 1.0)\n"
              << "  --D F             Cylinder diameter (default 1.0; sets Lx=Ly=10D)\n"
              << "  --xc F            Cylinder centre x (default 2.5*D)\n"
              << "  --yc F            Cylinder centre y (default 5.0*D)\n"
              << "  --np N            Marker count override (default 2 pi R / dx)\n"
              << "  --cfl F           CFL number, <=0 for fixed dt (default 0.5)\n"
              << "  --dt F            Fixed timestep when cfl<=0 (default 1e-3)\n"
              << "  --t_end F         End time (default 50)\n"
              << "  --diagnostics     Compute exact-solution error norms each step\n"
              << "  --profile         Write run_stats.json to run directory\n"
              << "  --help            Show this message\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        RunConfig cfg;
        cfg.nx    = 128;
        cfg.ny    = 128;
        cfg.re    = 200.0;
        cfg.cfl   = 0.5;
        cfg.t_end = 50.0;

        double D  = 1.0;
        double xc = -1.0, yc = -1.0;   // sentinel; resolved after D is finalised
        int    np = -1;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--nx") == 0 && i + 1 < argc)
                cfg.nx = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--ny") == 0 && i + 1 < argc)
                cfg.ny = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--re") == 0 && i + 1 < argc)
                cfg.re = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--u_inf") == 0 && i + 1 < argc)
                cfg.u_inf = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--D") == 0 && i + 1 < argc)
                D = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--xc") == 0 && i + 1 < argc)
                xc = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--yc") == 0 && i + 1 < argc)
                yc = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--np") == 0 && i + 1 < argc)
                np = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--cfl") == 0 && i + 1 < argc)
                cfg.cfl = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--dt") == 0 && i + 1 < argc)
                cfg.dt = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--t_end") == 0 && i + 1 < argc)
                cfg.t_end = std::atof(argv[++i]);
            else if (std::strcmp(argv[i], "--diagnostics") == 0)
                cfg.diagnostics = true;
            else if (std::strcmp(argv[i], "--profile") == 0)
                cfg.profile = true;
            else if (std::strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                Kokkos::finalize();
                return 0;
            }
        }

        // Domain: 10 D x 10 D, cylinder offset upstream so the wake fits.
        cfg.lx = 10.0 * D;
        cfg.ly = 10.0 * D;
        if (xc < 0.0) xc = 2.5 * D;
        if (yc < 0.0) yc = 5.0 * D;

        CylinderIB ib(xc, yc, 0.5 * D);
        ib.np_override = np;

        std::cout << "Immersed-boundary cylinder run\n"
                  << "  domain  : " << cfg.lx << " x " << cfg.ly << "\n"
                  << "  grid    : " << cfg.nx << " x " << cfg.ny << "\n"
                  << "  cylinder: D=" << D << ", center=(" << xc << ", " << yc << ")\n"
                  << "  Re      : " << cfg.re << "  (based on U*D)\n"
                  << "  u_inf   : " << cfg.u_inf << "\n";

        Solver<InflowOutflowBC, CrankThatNicolson, UniformStreamIC, CylinderIB> solver(
            cfg,
            InflowOutflowBC{cfg.u_inf},
            {},
            UniformStreamIC{cfg.u_inf},
            ib);

        std::cout << "  markers : " << solver.ib.np << "\n\n";

        solver.run();
    }
    Kokkos::finalize();
    return 0;
}
