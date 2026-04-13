#include "integrator.hpp"
#include "boundary_conditions.hpp"
#include "physics.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// ── ForwardEuler ─────────────────────────────────────────────────────────────

// A uniform velocity field (u=C, v=C) with periodic BC has zero convective and
// diffusive RHS, so ForwardEuler must leave u_star = u unchanged.
static void test_fe_uniform_field_unchanged() {
    constexpr int nx = 8, ny = 8;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid, 1);

    const double U = 2.0;
    auto u_h = Kokkos::create_mirror_view(state.u);
    auto v_h = Kokkos::create_mirror_view(state.v);
    for (int i = 0; i < (int)u_h.extent(0); ++i)
        for (int j = 0; j < (int)u_h.extent(1); ++j)
            u_h(i, j) = U;
    for (int i = 0; i < (int)v_h.extent(0); ++i)
        for (int j = 0; j < (int)v_h.extent(1); ++j)
            v_h(i, j) = U;
    Kokkos::deep_copy(state.u, u_h);
    Kokkos::deep_copy(state.v, v_h);

    PeriodicBC bc;
    bc.apply(state);   // refresh ghost cells before predict

    Kokkos::View<double**> u_star("u_star", grid.u_nx_total(), grid.u_ny_total());
    Kokkos::View<double**> v_star("v_star", grid.v_nx_total(), grid.v_ny_total());

    ForwardEuler fe;
    fe.predict(state, bc, u_star, v_star, /*re=*/100.0, /*dt=*/1e-3);

    auto u_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_star);
    auto v_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v_star);

    double max_err = 0.0;
    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i)
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            max_err = std::max(max_err, std::fabs(u_out(i, j) - U));
    for (int i = grid.v_i_begin(); i < grid.v_i_end(); ++i)
        for (int j = grid.v_j_begin(); j < grid.v_j_end(); ++j)
            max_err = std::max(max_err, std::fabs(v_out(i, j) - U));

    assert(max_err < 1e-10);
    std::cout << "PASS: test_fe_uniform_field_unchanged (max_err = " << max_err << ")\n";
}

// ── CrankThatNicolson ─────────────────────────────────────────────────────────

// A uniform velocity field (u=C, v=C) has zero RHS (conv and diff both vanish).
// The CN implicit solve for rhs=C and uniform initial guess must return u_star=C.
static void test_cn_uniform_field_unchanged() {
    constexpr int nx = 8, ny = 8;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid, 1);

    const double U = 1.5;
    auto u_h = Kokkos::create_mirror_view(state.u);
    auto v_h = Kokkos::create_mirror_view(state.v);
    for (int i = 0; i < (int)u_h.extent(0); ++i)
        for (int j = 0; j < (int)u_h.extent(1); ++j)
            u_h(i, j) = U;
    for (int i = 0; i < (int)v_h.extent(0); ++i)
        for (int j = 0; j < (int)v_h.extent(1); ++j)
            v_h(i, j) = U;
    Kokkos::deep_copy(state.u, u_h);
    Kokkos::deep_copy(state.v, v_h);

    PeriodicBC bc;
    bc.apply(state);

    Kokkos::View<double**> u_star("u_star", grid.u_nx_total(), grid.u_ny_total());
    Kokkos::View<double**> v_star("v_star", grid.v_nx_total(), grid.v_ny_total());

    CrankThatNicolson cn;
    cn.predict(state, bc, u_star, v_star, /*re=*/100.0, /*dt=*/1e-3);

    auto u_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_star);
    auto v_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v_star);

    double max_err = 0.0;
    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i)
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            max_err = std::max(max_err, std::fabs(u_out(i, j) - U));
    for (int i = grid.v_i_begin(); i < grid.v_i_end(); ++i)
        for (int j = grid.v_j_begin(); j < grid.v_j_end(); ++j)
            max_err = std::max(max_err, std::fabs(v_out(i, j) - U));

    assert(max_err < 1e-5);
    std::cout << "PASS: test_cn_uniform_field_unchanged (max_err = " << max_err << ")\n";
}

// A sinusoidal u field is diffused by CN.  The predicted u_star must have
// strictly lower kinetic energy than the initial u (diffusion dissipates).
// v = 0 throughout so the effect is purely from the u diffusion.
static void test_cn_diffusion_dissipates_energy() {
    constexpr int nx = 16, ny = 16;
    MacGrid2D grid(nx, ny, 2.0 * M_PI, 2.0 * M_PI);
    SimState state(grid, 1);

    // u = sin(x) on the u-face positions, v = 0
    auto u_h = Kokkos::create_mirror_view(state.u);
    for (int i = grid.u_i_begin(); i < grid.u_i_end(); ++i) {
        const double x = (i - grid.u_i_begin()) * grid.dx;  // face x-coordinate
        for (int j = grid.u_j_begin(); j < grid.u_j_end(); ++j)
            u_h(i, j) = std::sin(x);
    }
    Kokkos::deep_copy(state.u, u_h);

    PeriodicBC bc;
    bc.apply(state);

    const double ke_before = physics::compute_kinetic_energy(state);

    Kokkos::View<double**> u_star("u_star", grid.u_nx_total(), grid.u_ny_total());
    Kokkos::View<double**> v_star("v_star", grid.v_nx_total(), grid.v_ny_total());

    CrankThatNicolson cn;
    // Use a large dt and low Re so the diffusion effect is clearly visible
    cn.predict(state, bc, u_star, v_star, /*re=*/1.0, /*dt=*/0.1);

    // Temporarily copy u_star into state.u to reuse compute_kinetic_energy
    Kokkos::deep_copy(state.u, u_star);
    Kokkos::deep_copy(state.v, v_star);
    const double ke_after = physics::compute_kinetic_energy(state);

    assert(ke_after < ke_before);
    std::cout << "PASS: test_cn_diffusion_dissipates_energy"
              << " (ke_before=" << ke_before << ", ke_after=" << ke_after << ")\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        test_fe_uniform_field_unchanged();
        test_cn_uniform_field_unchanged();
        test_cn_diffusion_dissipates_energy();
    }
    Kokkos::finalize();
    std::cout << "All integrator tests passed.\n";
    return 0;
}
