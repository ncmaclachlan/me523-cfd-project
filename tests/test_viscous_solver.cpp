#include "viscous_solver.hpp"
#include "boundary_conditions.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// ViscousSolver solves (I - alpha*nabla^2) u = rhs via RBGS.
// For a uniform rhs=0 with periodic BC: solution should converge to zero
// (zero is the unique solution since I - alpha*lap is SPD and rhs=0).
static void test_zero_rhs_zero_solution() {
    constexpr int nx = 8, ny = 8;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid, 1);

    // u_star initialized to small nonzero values
    Kokkos::View<double**> u_star("u_star", grid.u_nx_total(), grid.u_ny_total());
    Kokkos::View<double**> rhs("rhs_u", grid.u_nx_total(), grid.u_ny_total());

    auto u_h = Kokkos::create_mirror_view(u_star);
    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i)
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            u_h(i, j) = 1.0;
    Kokkos::deep_copy(u_star, u_h);
    // rhs stays zero (Kokkos default-initializes to 0)

    PeriodicBC bc;
    ViscousSolver solver;
    solver.solve_u(state, bc, u_star, rhs, /*re=*/1.0, /*dt=*/1e-2);

    auto u_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_star);
    double max_err = 0.0;
    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i)
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            max_err = std::max(max_err, std::fabs(u_out(i, j)));

    assert(max_err < 1e-5);
    std::cout << "PASS: test_zero_rhs_zero_solution (max_err = " << max_err << ")\n";
}

// For rhs = constant C with periodic BC:
// nabla^2(C) = 0, so (I - alpha*lap)(C) = C.
// Therefore u_star = C is the exact solution.
static void test_constant_rhs_constant_solution() {
    constexpr int nx = 8, ny = 8;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid, 1);

    const double C = 2.5;
    Kokkos::View<double**> u_star("u_star", grid.u_nx_total(), grid.u_ny_total());
    Kokkos::View<double**> rhs("rhs_u", grid.u_nx_total(), grid.u_ny_total());

    // Initialize rhs = C everywhere (including ghost cells so BC fill is consistent)
    auto rhs_h = Kokkos::create_mirror_view(rhs);
    for (int i = 0; i < (int)rhs_h.extent(0); ++i)
        for (int j = 0; j < (int)rhs_h.extent(1); ++j)
            rhs_h(i, j) = C;
    Kokkos::deep_copy(rhs, rhs_h);

    // Initial guess: also C (gives the solver a head start but it must converge regardless)
    auto u_h = Kokkos::create_mirror_view(u_star);
    for (int i = 0; i < (int)u_h.extent(0); ++i)
        for (int j = 0; j < (int)u_h.extent(1); ++j)
            u_h(i, j) = C;
    Kokkos::deep_copy(u_star, u_h);

    PeriodicBC bc;
    ViscousSolver solver;
    solver.solve_u(state, bc, u_star, rhs, /*re=*/1.0, /*dt=*/1e-2);

    auto u_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_star);
    double max_err = 0.0;
    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i)
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            max_err = std::max(max_err, std::fabs(u_out(i, j) - C));

    assert(max_err < 1e-5);
    std::cout << "PASS: test_constant_rhs_constant_solution (max_err = " << max_err << ")\n";
}

// Symmetry check for solve_v: same logic as solve_u but for the v-stencil.
// rhs = 0 → v_star → 0.
static void test_solve_v_zero_rhs() {
    constexpr int nx = 8, ny = 8;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid, 1);

    Kokkos::View<double**> v_star("v_star", grid.v_nx_total(), grid.v_ny_total());
    Kokkos::View<double**> rhs("rhs_v", grid.v_nx_total(), grid.v_ny_total());

    auto v_h = Kokkos::create_mirror_view(v_star);
    for (int i = grid.v_i_begin(); i < grid.v_i_end(); ++i)
        for (int j = grid.v_j_begin(); j < grid.v_j_end(); ++j)
            v_h(i, j) = 1.0;
    Kokkos::deep_copy(v_star, v_h);

    PeriodicBC bc;
    ViscousSolver solver;
    solver.solve_v(state, bc, v_star, rhs, /*re=*/1.0, /*dt=*/1e-2);

    auto v_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v_star);
    double max_err = 0.0;
    for (int i = grid.v_i_begin(); i < grid.v_i_end(); ++i)
        for (int j = grid.v_j_begin(); j < grid.v_j_end(); ++j)
            max_err = std::max(max_err, std::fabs(v_out(i, j)));

    assert(max_err < 1e-5);
    std::cout << "PASS: test_solve_v_zero_rhs (max_err = " << max_err << ")\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        test_zero_rhs_zero_solution();
        test_constant_rhs_constant_solution();
        test_solve_v_zero_rhs();
    }
    Kokkos::finalize();
    std::cout << "All viscous solver tests passed.\n";
    return 0;
}
