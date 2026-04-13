#include "pressure_solver.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// Manufactured solution test:
//   Exact:  p(x,y) = sin(x) * sin(y)
//   RHS:    Lap(p)  = -2 * sin(x) * sin(y)
// on the periodic domain [0, 2*pi]^2.

static void test_manufactured_solution(int n, double& l2_error) {
    MacGrid2D grid(n, n, 2.0 * M_PI, 2.0 * M_PI);
    SimState state(grid);

    // Allocate RHS view matching pressure layout
    Kokkos::View<double**> rhs("rhs", grid.p_nx_total(), grid.p_ny_total());

    const double dx = grid.dx;
    const double dy = grid.dy;

    // Fill RHS = -2*sin(x)*sin(y) at cell centers
    auto rhs_h = Kokkos::create_mirror_view(rhs);
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            const double x = (i - 0.5) * dx;
            const double y = (j - 0.5) * dy;
            rhs_h(i, j) = -2.0 * std::sin(x) * std::sin(y);
        }
    }
    Kokkos::deep_copy(rhs, rhs_h);

    // Zero initial guess
    Kokkos::deep_copy(state.p, 0.0);

    PressureSolver solver;
    solver.tol = 1e-10;
    solver.max_vcycles = 200;
    auto result = solver.solve(state, rhs);

    std::cout << "  n=" << n
              << "  iters=" << result.iters
              << "  residual=" << result.final_residual << std::flush;

    // Check convergence
    assert(result.final_residual < 1e-8);
    assert(result.iters < 50);

    // Compute L2 error against exact solution
    auto p_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.p);
    double err2 = 0.0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            const double x = (i - 0.5) * dx;
            const double y = (j - 0.5) * dy;
            const double exact = std::sin(x) * std::sin(y);
            const double diff = p_h(i, j) - exact;
            err2 += diff * diff;
        }
    }
    l2_error = std::sqrt(err2 / (n * n));
    std::cout << "  L2_error=" << l2_error << std::endl;
}

static void test_zero_rhs() {
    const int n = 16;
    MacGrid2D grid(n, n, 2.0 * M_PI, 2.0 * M_PI);
    SimState state(grid);

    Kokkos::View<double**> rhs("rhs", grid.p_nx_total(), grid.p_ny_total());
    Kokkos::deep_copy(rhs, 0.0);
    Kokkos::deep_copy(state.p, 0.0);

    PressureSolver solver;
    auto result = solver.solve(state, rhs);

    // With zero RHS and zero initial guess, solution should remain zero
    auto p_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.p);
    double max_abs = 0.0;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            max_abs = std::max(max_abs, std::abs(p_h(i, j)));

    std::cout << "  zero_rhs: max|p| = " << max_abs << std::endl;
    assert(max_abs < 1e-12);
}

static void test_convergence_order() {
    // Check second-order convergence: error(2h)/error(h) ~ 4
    double err_16, err_32, err_64;

    std::cout << "Grid refinement study:" << std::endl;
    test_manufactured_solution(16, err_16);
    test_manufactured_solution(32, err_32);
    test_manufactured_solution(64, err_64);

    const double rate_1 = std::log2(err_16 / err_32);
    const double rate_2 = std::log2(err_32 / err_64);

    std::cout << "  rate(16->32) = " << rate_1 << std::endl;
    std::cout << "  rate(32->64) = " << rate_2 << std::endl;

    // Expect ~2nd order (allow some margin)
    assert(rate_1 > 1.8);
    assert(rate_2 > 1.8);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << "=== test_zero_rhs ===" << std::endl;
        test_zero_rhs();

        std::cout << "=== test_convergence_order ===" << std::endl;
        test_convergence_order();

        std::cout << "All pressure solver tests passed." << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
