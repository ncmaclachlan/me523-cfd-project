#include "physics.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// Uniform velocity: u=2, v=3 on a 4x4 grid
// KE = 0.5 * (n_u_owned * 4 + n_v_owned * 9)
//    = 0.5 * (5*4 * 4 + 4*5 * 9)
//    = 0.5 * (80 + 180) = 130
static void test_uniform_field() {
    constexpr int nx = 4, ny = 4;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid, 1);

    auto u_h = Kokkos::create_mirror_view(state.u);
    auto v_h = Kokkos::create_mirror_view(state.v);

    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i)
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            u_h(i, j) = 2.0;

    for (int i = grid.v_i_begin(); i < grid.v_i_end(); ++i)
        for (int j = grid.v_j_begin(); j < grid.v_j_end(); ++j)
            v_h(i, j) = 3.0;

    Kokkos::deep_copy(state.u, u_h);
    Kokkos::deep_copy(state.v, v_h);

    double ke = physics::compute_kinetic_energy(state);

    // u owned: (nx+1)*ny = 5*4 = 20 entries, each 2^2=4  -> sum_u2 = 80
    // v owned: nx*(ny+1) = 4*5 = 20 entries, each 3^2=9  -> sum_v2 = 180
    // KE = 0.5*(sum_u2 + sum_v2) * dx*dy  — compute_kinetic_energy integrates
    // over the domain, not a plain discrete sum. Area factor added in commit
    // f77dcf9; expected value updated here to match (dx=dy=0.25 for 4x4 on [0,1]^2).
    double expected = 130.0 * grid.dx * grid.dy;
    assert(std::fabs(ke - expected) < 1e-10);
    std::cout << "PASS: test_uniform_field (KE = " << ke << ")\n";
}

// Zero velocity field should give KE = 0
static void test_zero_field() {
    MacGrid2D grid(3, 3, 1.0, 1.0);
    SimState state(grid, 1);
    // Views are zero-initialized by Kokkos

    double ke = physics::compute_kinetic_energy(state);
    assert(std::fabs(ke) < 1e-15);
    std::cout << "PASS: test_zero_field (KE = " << ke << ")\n";
}

// Single interior point with known value on a 1x1 grid
// u has 2x1 owned entries, v has 1x2 owned entries
static void test_single_cell() {
    MacGrid2D grid(1, 1, 1.0, 1.0);
    SimState state(grid, 1);

    auto u_h = Kokkos::create_mirror_view(state.u);
    auto v_h = Kokkos::create_mirror_view(state.v);

    // Set all owned u faces to 1.0
    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i)
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            u_h(i, j) = 1.0;

    // Set all owned v faces to 1.0
    for (int i = grid.v_i_begin(); i < grid.v_i_end(); ++i)
        for (int j = grid.v_j_begin(); j < grid.v_j_end(); ++j)
            v_h(i, j) = 1.0;

    Kokkos::deep_copy(state.u, u_h);
    Kokkos::deep_copy(state.v, v_h);

    double ke = physics::compute_kinetic_energy(state);

    // u: (1+1)*1 = 2 entries of 1^2  -> sum_u2 = 2
    // v: 1*(1+1) = 2 entries of 1^2  -> sum_v2 = 2
    // KE = 0.5*(2+2) = 2.0
    double expected = 2.0;
    assert(std::fabs(ke - expected) < 1e-10);
    std::cout << "PASS: test_single_cell (KE = " << ke << ")\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        test_zero_field();
        test_uniform_field();
        test_single_cell();
    }
    Kokkos::finalize();
    std::cout << "All kinetic energy tests passed.\n";
    return 0;
}
