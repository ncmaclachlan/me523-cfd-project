#pragma once
#include "sim_state.hpp"
#include <Kokkos_Core.hpp>

struct LidDrivenCavityBC {
    double lid_velocity = 1.0;

    LidDrivenCavityBC() = default;
    LidDrivenCavityBC(double u_lid) : lid_velocity(u_lid) {}

    void apply_u(const MacGrid2D& g, Kokkos::View<double**> u) const {
        const double u_lid = lid_velocity;

        // Left/right walls: u-face sits exactly on the wall, enforce no-slip directly
        Kokkos::parallel_for("ldc_u_lr",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(int j) {
                u(g.u_i_begin(),     j) = 0.0;  // left wall
                u(g.u_i_end() - 1,   j) = 0.0;  // right wall
            });

        // Bottom/top: u-face is half a cell from the wall, enforce via ghost cell
        //   bottom (y=0, no-slip): u_ghost = -u_interior  -> avg = 0 at wall
        //   top    (y=Ly, lid):    u_ghost = 2*U_lid - u_interior
        Kokkos::parallel_for("ldc_u_tb",
            Kokkos::RangePolicy<>(g.u_i_begin(), g.u_i_end()),
            KOKKOS_LAMBDA(int i) {
                u(i, g.u_j_begin() - 1) = -u(i, g.u_j_begin());
                u(i, g.u_j_end())       = 2.0 * u_lid - u(i, g.u_j_end() - 1);
            });
    }

    void apply_v(const MacGrid2D& g, Kokkos::View<double**> v) const {
        // Bottom/top walls: v-face sits exactly on the wall, enforce no-slip directly
        Kokkos::parallel_for("ldc_v_tb",
            Kokkos::RangePolicy<>(g.v_i_begin(), g.v_i_end()),
            KOKKOS_LAMBDA(int i) {
                v(i, g.v_j_begin())     = 0.0;  // bottom wall
                v(i, g.v_j_end() - 1)   = 0.0;  // top wall
            });

        // Left/right: v-face is half a cell from the wall, enforce via ghost cell
        //   left  (x=0,  no-slip): v_ghost = -v_interior
        //   right (x=Lx, no-slip): v_ghost = -v_interior
        Kokkos::parallel_for("ldc_v_lr",
            Kokkos::RangePolicy<>(g.v_j_begin(), g.v_j_end()),
            KOKKOS_LAMBDA(int j) {
                v(g.v_i_begin() - 1, j) = -v(g.v_i_begin(), j);
                v(g.v_i_end(),       j) = -v(g.v_i_end() - 1, j);
            });
    }

    void apply(SimState& s) const {
        apply_u(s.grid, s.u);
        apply_v(s.grid, s.v);
    }
};

struct PeriodicBC {
    void apply_u(const MacGrid2D& g, Kokkos::View<double**> u) const {
        Kokkos::parallel_for("periodic_u_x",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(const int j) {
                u(g.u_i_begin() - 1, j) = u(g.u_i_end() - 1, j);
                u(g.u_i_end(),       j) = u(g.u_i_begin(),   j);
            });

        Kokkos::parallel_for("periodic_u_y",
            Kokkos::RangePolicy<>(g.u_i_begin(), g.u_i_end()),
            KOKKOS_LAMBDA(const int i) {
                u(i, g.u_j_begin() - 1) = u(i, g.u_j_end() - 1);
                u(i, g.u_j_end())       = u(i, g.u_j_begin());
            });

        Kokkos::parallel_for("periodic_u_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                u(g.u_i_begin() - 1, g.u_j_begin() - 1) = u(g.u_i_end() - 1, g.u_j_end() - 1);
                u(g.u_i_begin() - 1, g.u_j_end())       = u(g.u_i_end() - 1, g.u_j_begin());
                u(g.u_i_end(),       g.u_j_begin() - 1) = u(g.u_i_begin(),   g.u_j_end() - 1);
                u(g.u_i_end(),       g.u_j_end())       = u(g.u_i_begin(),   g.u_j_begin());
            });
    }

    void apply_v(const MacGrid2D& g, Kokkos::View<double**> v) const {
        Kokkos::parallel_for("periodic_v_x",
            Kokkos::RangePolicy<>(g.v_j_begin(), g.v_j_end()),
            KOKKOS_LAMBDA(const int j) {
                v(g.v_i_begin() - 1, j) = v(g.v_i_end() - 1, j);
                v(g.v_i_end(),       j) = v(g.v_i_begin(),   j);
            });

        Kokkos::parallel_for("periodic_v_y",
            Kokkos::RangePolicy<>(g.v_i_begin(), g.v_i_end()),
            KOKKOS_LAMBDA(const int i) {
                v(i, g.v_j_begin() - 1) = v(i, g.v_j_end() - 1);
                v(i, g.v_j_end())       = v(i, g.v_j_begin());
            });

        Kokkos::parallel_for("periodic_v_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                v(g.v_i_begin() - 1, g.v_j_begin() - 1) = v(g.v_i_end() - 1, g.v_j_end() - 1);
                v(g.v_i_begin() - 1, g.v_j_end())       = v(g.v_i_end() - 1, g.v_j_begin());
                v(g.v_i_end(),       g.v_j_begin() - 1) = v(g.v_i_begin(),   g.v_j_end() - 1);
                v(g.v_i_end(),       g.v_j_end())       = v(g.v_i_begin(),   g.v_j_begin());
            });
    }

    void apply(SimState& s) const {
        const MacGrid2D& g = s.grid;

        // pressure
        Kokkos::parallel_for("periodic_p_x",
            Kokkos::RangePolicy<>(g.p_j_begin(), g.p_j_end()),
            KOKKOS_LAMBDA(const int j) {
                s.p(g.p_i_begin() - 1, j) = s.p(g.p_i_end() - 1, j);
                s.p(g.p_i_end(),       j) = s.p(g.p_i_begin(),   j);
            });

        Kokkos::parallel_for("periodic_p_y",
            Kokkos::RangePolicy<>(g.p_i_begin(), g.p_i_end()),
            KOKKOS_LAMBDA(const int i) {
                s.p(i, g.p_j_begin() - 1) = s.p(i, g.p_j_end() - 1);
                s.p(i, g.p_j_end())       = s.p(i, g.p_j_begin());
            });

        Kokkos::parallel_for("periodic_p_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                s.p(g.p_i_begin() - 1, g.p_j_begin() - 1) = s.p(g.p_i_end() - 1, g.p_j_end() - 1);
                s.p(g.p_i_begin() - 1, g.p_j_end())       = s.p(g.p_i_end() - 1, g.p_j_begin());
                s.p(g.p_i_end(),       g.p_j_begin() - 1) = s.p(g.p_i_begin(),   g.p_j_end() - 1);
                s.p(g.p_i_end(),       g.p_j_end())       = s.p(g.p_i_begin(),   g.p_j_begin());
            });

        apply_u(g, s.u);
        apply_v(g, s.v);
    }
};
