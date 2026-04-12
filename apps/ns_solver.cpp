#include <Kokkos_Core.hpp>
#include "problem_traits.hpp"
#include "solver.hpp"

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        using Traits = HW7;

        Solver<Traits> solver(
            MacGrid2D(64, 64, 1.0, 1.0),
            IncompressibleNS{100.0},
            LidDrivenCavityBC{1.0}
        );

        const double dt    = 1e-3;
        const double t_end = 1.0;
        Traits::Output output("output.csv");

        while (solver.state.time < t_end) {
            solver.advance(dt);
        }

        output.write(solver.state);
    }
    Kokkos::finalize();
    return 0;
}
