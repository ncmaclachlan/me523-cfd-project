#include <Kokkos_Core.hpp>
#include "solver.hpp"

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        RunConfig cfg;
        cfg.nx = 64;
        cfg.ny = 64;
        cfg.re = 100.0;
        cfg.dt = 1e-5;
        cfg.t_end = 1.0;
=======
        cfg.nx    = 64;
        cfg.ny    = 64;
        cfg.re    = 100.0;
        cfg.cfl   = 0.5;
        cfg.t_end = 10.0;
>>>>>>> main

        Solver<PeriodicBC, CrankThatNicolson> solver(cfg);
        solver.run();
    }
    Kokkos::finalize();
    return 0;
}
