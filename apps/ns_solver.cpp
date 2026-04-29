#include <Kokkos_Core.hpp>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include "solver.hpp"

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --nx N            Grid cells in x (default 64)\n"
              << "  --ny N            Grid cells in y (default 64)\n"
              << "  --re F            Reynolds number  (default 100)\n"
              << "  --cfl F           CFL number, <=0 for fixed dt (default 0.5)\n"
              << "  --dt F            Fixed timestep when cfl<=0 (default 1e-3)\n"
              << "  --t_end F         End time (default 10)\n"
              << "  --diagnostics     Compute exact-solution error norms each step\n"
              << "  --profile         Write run_stats.json to run directory\n"
              << "  --help            Show this message\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        RunConfig cfg;
        cfg.nx    = 64;
        cfg.ny    = 64;
        cfg.re    = 100.0;
        cfg.cfl   = 0.5;
        cfg.t_end = 10.0;

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--nx") == 0 && i + 1 < argc)
                cfg.nx = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--ny") == 0 && i + 1 < argc)
                cfg.ny = std::atoi(argv[++i]);
            else if (std::strcmp(argv[i], "--re") == 0 && i + 1 < argc)
                cfg.re = std::atof(argv[++i]);
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

        Solver<PeriodicBC, CrankThatNicolson, TaylorGreenIC> solver(cfg);
        solver.run();
    }
    Kokkos::finalize();
    return 0;
}
