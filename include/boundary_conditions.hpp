#pragma once
#include "sim_state.hpp"
#include <Kokkos_Core.hpp>

struct LidDrivenCavityBC {
    double lid_velocity = 1.0;

    LidDrivenCavityBC() = default;
    LidDrivenCavityBC(double u_lid) : lid_velocity(u_lid) {}

    void apply(SimState& s) const {
        // no-slip on left/right/bottom walls, lid velocity on top boundary
        // not implemented here
    }
};

struct PeriodicBC {
    void apply(SimState& s) const {
        const MacGrid2D& g = s.grid;

        SimState::View2D p = s.p;
        SimState::View2D u = s.u;
        SimState::View2D v = s.v;

        // ------------------------------------------------------------------
        // p: periodic wrap in x
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_p_x",
            Kokkos::RangePolicy<>(g.p_j_begin(), g.p_j_end()),
            KOKKOS_LAMBDA(const int j) {
                p(g.p_i_begin() - 1, j) = p(g.p_i_end() - 1, j);
                p(g.p_i_end(),     j)   = p(g.p_i_begin(),   j);
            });

        // ------------------------------------------------------------------
        // p: periodic wrap in y
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_p_y",
            Kokkos::RangePolicy<>(g.p_i_begin(), g.p_i_end()),
            KOKKOS_LAMBDA(const int i) {
                p(i, g.p_j_begin() - 1) = p(i, g.p_j_end() - 1);
                p(i, g.p_j_end())       = p(i, g.p_j_begin());
            });

        // ------------------------------------------------------------------
        // p corners
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_p_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                p(g.p_i_begin() - 1, g.p_j_begin() - 1) = p(g.p_i_end() - 1, g.p_j_end() - 1);
                p(g.p_i_begin() - 1, g.p_j_end())       = p(g.p_i_end() - 1, g.p_j_begin());
                p(g.p_i_end(),       g.p_j_begin() - 1) = p(g.p_i_begin(),   g.p_j_end() - 1);
                p(g.p_i_end(),       g.p_j_end())       = p(g.p_i_begin(),   g.p_j_begin());
            });

        // ------------------------------------------------------------------
        // u: periodic wrap in x
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_u_x",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(const int j) {
                u(g.u_i_begin() - 1, j) = u(g.u_i_end() - 1, j);
                u(g.u_i_end(),     j)   = u(g.u_i_begin(),   j);
            });

        // ------------------------------------------------------------------
        // u: periodic wrap in y
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_u_y",
            Kokkos::RangePolicy<>(g.u_i_begin(), g.u_i_end()),
            KOKKOS_LAMBDA(const int i) {
                u(i, g.u_j_begin() - 1) = u(i, g.u_j_end() - 1);
                u(i, g.u_j_end())       = u(i, g.u_j_begin());
            });

        // ------------------------------------------------------------------
        // u corners
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_u_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                u(g.u_i_begin() - 1, g.u_j_begin() - 1) = u(g.u_i_end() - 1, g.u_j_end() - 1);
                u(g.u_i_begin() - 1, g.u_j_end())       = u(g.u_i_end() - 1, g.u_j_begin());
                u(g.u_i_end(),       g.u_j_begin() - 1) = u(g.u_i_begin(),   g.u_j_end() - 1);
                u(g.u_i_end(),       g.u_j_end())       = u(g.u_i_begin(),   g.u_j_begin());
            });

        // ------------------------------------------------------------------
        // v: periodic wrap in x
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_v_x",
            Kokkos::RangePolicy<>(g.v_j_begin(), g.v_j_end()),
            KOKKOS_LAMBDA(const int j) {
                v(g.v_i_begin() - 1, j) = v(g.v_i_end() - 1, j);
                v(g.v_i_end(),     j)   = v(g.v_i_begin(),   j);
            });

        // ------------------------------------------------------------------
        // v: periodic wrap in y
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_v_y",
            Kokkos::RangePolicy<>(g.v_i_begin(), g.v_i_end()),
            KOKKOS_LAMBDA(const int i) {
                v(i, g.v_j_begin() - 1) = v(i, g.v_j_end() - 1);
                v(i, g.v_j_end())       = v(i, g.v_j_begin());
            });

        // ------------------------------------------------------------------
        // v corners
        // ------------------------------------------------------------------
        Kokkos::parallel_for(
            "periodic_v_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                v(g.v_i_begin() - 1, g.v_j_begin() - 1) = v(g.v_i_end() - 1, g.v_j_end() - 1);
                v(g.v_i_begin() - 1, g.v_j_end())       = v(g.v_i_end() - 1, g.v_j_begin());
                v(g.v_i_end(),       g.v_j_begin() - 1) = v(g.v_i_begin(),   g.v_j_end() - 1);
                v(g.v_i_end(),       g.v_j_end())       = v(g.v_i_begin(),   g.v_j_begin());
            });

    }
};