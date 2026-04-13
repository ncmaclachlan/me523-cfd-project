#include "physics.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// Uniform velocity field: du/dx = 0, dv/dy = 0 everywhere -> div = 0
static void test_uniform_field() {
    constexpr int nx = 4, ny = 4;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid, 1);

    auto u_h = Kokkos::create_mirror_view(state.u);
    auto v_h = Kokkos::create_mirror_view(state.v);

    for (int i = 0; i < grid.u_nx_total(); ++i)
        for (int j = 0; j < grid.u_ny_total(); ++j)
            u_h(i, j) = 2.0;

    for (int i = 0; i < grid.v_nx_total(); ++i)
        for (int j = 0; j < grid.v_ny_total(); ++j)
            v_h(i, j) = 3.0;

    Kokkos::deep_copy(state.u, u_h);
    Kokkos::deep_copy(state.v, v_h);

    double l2 = physics::compute_l2_divergence(state);
    assert(std::fabs(l2) < 1e-12);
    std::cout << "PASS: test_uniform_field (L2_div = " << l2 << ")\n";
}

// Zero velocity field -> div = 0
static void test_zero_field() {
    MacGrid2D grid(3, 3, 1.0, 1.0);
    SimState state(grid, 1);

    double l2 = physics::compute_l2_divergence(state);
    assert(std::fabs(l2) < 1e-15);
    std::cout << "PASS: test_zero_field (L2_div = " << l2 << ")\n";
}

// Linear velocity u = x, v = -y  -> du/dx = 1, dv/dy = -1  -> div = 0
// On MAC grid with dx = lx/nx:
//   u on left face of cell i is at x = (i - ng) * dx
//   so u(i,j) = (i - ng) * dx, and u(i+1,j) - u(i,j) = dx  -> du/dx = 1
//   v on bottom face of cell j is at y = (j - ng) * dy
//   so v(i,j) = -(j - ng) * dy, and v(i,j+1) - v(i,j) = -dy -> dv/dy = -1
static void test_divergence_free_linear() {
    constexpr int nx = 5, ny = 5;
    constexpr double lx = 2.0, ly = 2.0;
    MacGrid2D grid(nx, ny, lx, ly);
    SimState state(grid, 1);

    auto u_h = Kokkos::create_mirror_view(state.u);
    auto v_h = Kokkos::create_mirror_view(state.v);

    double dx = grid.dx;
    double dy = grid.dy;
    int ng = MacGrid2D::ng;

    for (int i = 0; i < grid.u_nx_total(); ++i)
        for (int j = 0; j < grid.u_ny_total(); ++j)
            u_h(i, j) = (i - ng) * dx;

    for (int i = 0; i < grid.v_nx_total(); ++i)
        for (int j = 0; j < grid.v_ny_total(); ++j)
            v_h(i, j) = -(j - ng) * dy;

    Kokkos::deep_copy(state.u, u_h);
    Kokkos::deep_copy(state.v, v_h);

    double l2 = physics::compute_l2_divergence(state);
    assert(std::fabs(l2) < 1e-12);
    std::cout << "PASS: test_divergence_free_linear (L2_div = " << l2 << ")\n";
}

// Known nonzero divergence: u = x, v = y  -> du/dx = 1, dv/dy = 1  -> div = 2 everywhere
// RMS = sqrt(sum(div^2) / (nx*ny)) = sqrt(4) = 2
static void test_known_divergence() {
    constexpr int nx = 4, ny = 4;
    constexpr double lx = 1.0, ly = 1.0;
    MacGrid2D grid(nx, ny, lx, ly);
    SimState state(grid, 1);

    auto u_h = Kokkos::create_mirror_view(state.u);
    auto v_h = Kokkos::create_mirror_view(state.v);

    double dx = grid.dx;
    double dy = grid.dy;
    int ng = MacGrid2D::ng;

    for (int i = 0; i < grid.u_nx_total(); ++i)
        for (int j = 0; j < grid.u_ny_total(); ++j)
            u_h(i, j) = (i - ng) * dx;

    for (int i = 0; i < grid.v_nx_total(); ++i)
        for (int j = 0; j < grid.v_ny_total(); ++j)
            v_h(i, j) = (j - ng) * dy;

    Kokkos::deep_copy(state.u, u_h);
    Kokkos::deep_copy(state.v, v_h);

    double l2 = physics::compute_l2_divergence(state);

    // Each cell has div = 1 + 1 = 2, so sum(div^2) = nx*ny * 4
    // RMS = sqrt(nx*ny * 4 / (nx*ny)) = 2
    double expected = 2.0;
    assert(std::fabs(l2 - expected) < 1e-10);
    std::cout << "PASS: test_known_divergence (L2_div = " << l2
              << ", expected = " << expected << ")\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        test_zero_field();
        test_uniform_field();
        test_divergence_free_linear();
        test_known_divergence();
    }
    Kokkos::finalize();
    std::cout << "All L2 divergence tests passed.\n";
    return 0;
}
