#pragma once
#include <Kokkos_Core.hpp>
#include <cmath>
#include "sim_state.hpp"

// Direct-forcing immersed boundary (Uhlmann 2005) for a stationary cylinder.
//
// Pipeline per timestep, applied to u* between predictor and pressure RHS:
//   1. interpolate u* from staggered grid to each marker via 3x3 Roma-kernel stencil
//   2. F = (u_IB - U)/dt  (u_IB = 0 for stationary body)
//   3. spread F back to the grid with the same kernel
//   4. u** = u* + dt * f
//
// Reference: M. Uhlmann, J. Comp. Phys. 209 (2005) 448-476.

KOKKOS_INLINE_FUNCTION
double ib_roma_1d(double r, double dx) {
    // Roma kernel (Eq. 48 of project notes), returns weight per unit length.
    const double a = Kokkos::fabs(r) / dx;
    if (a <= 0.5) {
        return (1.0 / (3.0 * dx))
             * (1.0 + Kokkos::sqrt(-3.0 * a * a + 1.0));
    } else if (a <= 1.5) {
        return (1.0 / (6.0 * dx))
             * (5.0 - 3.0 * a
                - Kokkos::sqrt(-3.0 * (1.0 - a) * (1.0 - a) + 1.0));
    }
    return 0.0;
}

struct NoIB {
    void apply(SimState& /*s*/,
               Kokkos::View<double**> /*u_star*/,
               Kokkos::View<double**> /*v_star*/,
               double /*dt*/) const {}
};

struct CylinderIB {
    using View1D = Kokkos::View<double*>;

    double xc = 0.0, yc = 0.0, R = 0.5;   // physical cylinder geometry
    int    np_override = -1;               // user override; <=0 picks 2 pi R / dx
    int    np = 0;                         // number of markers (computed in init)
    double Vp = 0.0;                       // surface element area (uniform)

    View1D Xp, Yp;
    View1D Fxp, Fyp;
    View1D Ux, Uy;  // interpolated marker velocities

    CylinderIB() = default;
    CylinderIB(double xc_, double yc_, double R_) : xc(xc_), yc(yc_), R(R_) {}

    // Allocate and seed markers uniformly around the circumference.
    // Np ~ 2 pi R / dx  so there is roughly one marker per cell.
    void init(const MacGrid2D& g) {
        const double dx = g.dx;
        np = (np_override > 0)
             ? np_override
             : static_cast<int>(std::ceil(2.0 * M_PI * R / dx));
        if (np < 4) np = 4;

        Vp  = 2.0 * M_PI * R * dx / static_cast<double>(np);
        Xp  = View1D("ib_Xp",  np);
        Yp  = View1D("ib_Yp",  np);
        Fxp = View1D("ib_Fxp", np);
        Fyp = View1D("ib_Fyp", np);
        Ux  = View1D("ib_Ux",  np);
        Uy  = View1D("ib_Uy",  np);

        const double xc_l = xc, yc_l = yc, R_l = R;
        const int    np_l = np;
        auto Xp_l = Xp;
        auto Yp_l = Yp;
        Kokkos::parallel_for("ib_seed_markers",
            Kokkos::RangePolicy<>(0, np),
            KOKKOS_LAMBDA(int p) {
                const double theta = 2.0 * M_PI * static_cast<double>(p)
                                   / static_cast<double>(np_l);
                Xp_l(p) = xc_l + R_l * Kokkos::cos(theta);
                Yp_l(p) = yc_l + R_l * Kokkos::sin(theta);
            });
        Kokkos::fence();
    }

    void apply(SimState& s,
               Kokkos::View<double**> u_star,
               Kokkos::View<double**> v_star,
               double dt) const {
        const MacGrid2D& g = s.grid;
        const double dx = g.dx;
        const double dy = g.dy;
        const int    ng = MacGrid2D::ng;
        const double Vc = dx * dy;
        const double Vp_l = Vp;
        const int    np_l = np;

        // Capture views by value so KOKKOS_LAMBDA sees them.
        auto Xp_l  = Xp;
        auto Yp_l  = Yp;
        auto Ux_l  = Ux;
        auto Uy_l  = Uy;
        auto Fxp_l = Fxp;
        auto Fyp_l = Fyp;

        // --- 1. interpolate u_star, v_star to each marker ---
        Kokkos::parallel_for("ib_interp",
            Kokkos::RangePolicy<>(0, np_l),
            KOKKOS_LAMBDA(int p) {
                const double xp = Xp_l(p);
                const double yp = Yp_l(p);

                // Nearest u-face: x_u(i)=(i-ng)*dx, y_u(j)=(j-ng+0.5)*dy.
                const int i_uc = static_cast<int>(Kokkos::floor(xp / dx + 0.5)) + ng;
                const int j_uc = static_cast<int>(Kokkos::floor(yp / dy))      + ng;
                double sum_u = 0.0;
                for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj) {
                    const int i = i_uc + di;
                    const int j = j_uc + dj;
                    const double xf = (i - ng) * dx;
                    const double yf = (j - ng + 0.5) * dy;
                    const double w = ib_roma_1d(xf - xp, dx)
                                   * ib_roma_1d(yf - yp, dy);
                    sum_u += u_star(i, j) * w * Vc;
                }
                Ux_l(p) = sum_u;

                // Nearest v-face: x_v(i)=(i-ng+0.5)*dx, y_v(j)=(j-ng)*dy.
                const int i_vc = static_cast<int>(Kokkos::floor(xp / dx))      + ng;
                const int j_vc = static_cast<int>(Kokkos::floor(yp / dy + 0.5)) + ng;
                double sum_v = 0.0;
                for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj) {
                    const int i = i_vc + di;
                    const int j = j_vc + dj;
                    const double xf = (i - ng + 0.5) * dx;
                    const double yf = (j - ng) * dy;
                    const double w = ib_roma_1d(xf - xp, dx)
                                   * ib_roma_1d(yf - yp, dy);
                    sum_v += v_star(i, j) * w * Vc;
                }
                Uy_l(p) = sum_v;
            });

        // --- 2. compute marker forces (u_IB = 0 for stationary cylinder) ---
        const double inv_dt = 1.0 / dt;
        Kokkos::parallel_for("ib_force",
            Kokkos::RangePolicy<>(0, np_l),
            KOKKOS_LAMBDA(int p) {
                Fxp_l(p) = (0.0 - Ux_l(p)) * inv_dt;
                Fyp_l(p) = (0.0 - Uy_l(p)) * inv_dt;
            });

        // --- 3. spread forces back to grid and add dt*f to u_star/v_star ---
        Kokkos::parallel_for("ib_spread_u",
            Kokkos::RangePolicy<>(0, np_l),
            KOKKOS_LAMBDA(int p) {
                const double xp = Xp_l(p);
                const double yp = Yp_l(p);
                const double Fxp_p = Fxp_l(p);

                const int i_uc = static_cast<int>(Kokkos::floor(xp / dx + 0.5)) + ng;
                const int j_uc = static_cast<int>(Kokkos::floor(yp / dy))      + ng;
                for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj) {
                    const int i = i_uc + di;
                    const int j = j_uc + dj;
                    const double xf = (i - ng) * dx;
                    const double yf = (j - ng + 0.5) * dy;
                    const double w = ib_roma_1d(xf - xp, dx)
                                   * ib_roma_1d(yf - yp, dy);
                    Kokkos::atomic_add(&u_star(i, j), dt * Fxp_p * w * Vp_l);
                }
            });

        Kokkos::parallel_for("ib_spread_v",
            Kokkos::RangePolicy<>(0, np_l),
            KOKKOS_LAMBDA(int p) {
                const double xp = Xp_l(p);
                const double yp = Yp_l(p);
                const double Fyp_p = Fyp_l(p);

                const int i_vc = static_cast<int>(Kokkos::floor(xp / dx))      + ng;
                const int j_vc = static_cast<int>(Kokkos::floor(yp / dy + 0.5)) + ng;
                for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj) {
                    const int i = i_vc + di;
                    const int j = j_vc + dj;
                    const double xf = (i - ng + 0.5) * dx;
                    const double yf = (j - ng) * dy;
                    const double w = ib_roma_1d(xf - xp, dx)
                                   * ib_roma_1d(yf - yp, dy);
                    Kokkos::atomic_add(&v_star(i, j), dt * Fyp_p * w * Vp_l);
                }
            });
    }
};
