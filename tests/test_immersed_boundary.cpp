#include "immersed_boundary.hpp"
#include "sim_state.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// After one IB application of a stationary cylinder in a uniform stream,
// the velocity sampled at marker locations should be reduced toward zero
// (specifically: u at markers = U^* + dt*F = U_IB = 0 in the
// continuous limit; with discrete spreading the residual is O(1) of the
// inflow when only one application is done, but should drop dramatically
// after a single forcing pass for points strictly inside the kernel
// support of all 9 stencil weights).
static void test_force_drives_marker_velocity_toward_zero() {
    constexpr int nx = 64, ny = 64;
    constexpr double lx = 5.0, ly = 5.0;
    MacGrid2D grid(nx, ny, lx, ly);
    SimState state(grid, 1);

    Kokkos::View<double**> u_star("u_star", grid.u_nx_total(), grid.u_ny_total());
    Kokkos::View<double**> v_star("v_star", grid.v_nx_total(), grid.v_ny_total());

    // Uniform stream in u-direction.
    Kokkos::deep_copy(u_star, 1.0);
    Kokkos::deep_copy(v_star, 0.0);

    CylinderIB ib(2.5, 2.5, 0.5);
    ib.init(grid);
    assert(ib.np >= 4);

    const double dt = 1e-2;
    ib.apply(state, u_star, v_star, dt);

    // Re-interpolate u_star at markers; should now be near zero.
    auto Xp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ib.Xp);
    auto Yp_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ib.Yp);
    auto u_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, u_star);

    const double dx = grid.dx, dy = grid.dy;
    const int    ng = MacGrid2D::ng;

    double max_u_at_markers = 0.0;
    for (int p = 0; p < ib.np; ++p) {
        const double xp = Xp_h(p), yp = Yp_h(p);
        const int i_uc = static_cast<int>(std::floor(xp / dx + 0.5)) + ng;
        const int j_uc = static_cast<int>(std::floor(yp / dy))       + ng;
        double sum = 0.0;
        for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj) {
            const int i = i_uc + di;
            const int j = j_uc + dj;
            const double xf = (i - ng) * dx;
            const double yf = (j - ng + 0.5) * dy;
            const double w = ib_roma_1d(xf - xp, dx)
                           * ib_roma_1d(yf - yp, dy);
            sum += u_h(i, j) * w * dx * dy;
        }
        if (std::fabs(sum) > max_u_at_markers) max_u_at_markers = std::fabs(sum);
    }

    std::cout << "max |u| at markers after one IB pass: "
              << max_u_at_markers << " (was 1.0 before)\n";
    // Direct forcing reduces the inflow velocity at the markers; the residual
    // is non-zero because spread+interp is not an exact inverse of itself
    // (the forcing magnitude scales by sum_w^2, not sum_w). For a single pass
    // we expect the residual to be substantially less than the inflow value.
    assert(max_u_at_markers < 0.6);
}

static void test_roma_kernel_partition_of_unity() {
    // sum over a fine grid of the 1D Roma kernel should approximately integrate to 1.
    const double dx = 1.0;
    double sum = 0.0;
    const int N = 2000;
    const double dr = 4.0 / N;
    for (int k = 0; k < N; ++k) {
        const double r = -2.0 + (k + 0.5) * dr;
        sum += ib_roma_1d(r, dx) * dr;
    }
    std::cout << "Roma kernel integral: " << sum << " (expected 1)\n";
    assert(std::fabs(sum - 1.0) < 1e-3);
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        test_roma_kernel_partition_of_unity();
        test_force_drives_marker_velocity_toward_zero();
    }
    Kokkos::finalize();
    std::cout << "All IB tests passed.\n";
    return 0;
}
