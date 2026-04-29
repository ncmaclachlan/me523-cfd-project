#pragma once
#include <Kokkos_Core.hpp>
#include "sim_state.hpp"

struct ZeroIC {
    void apply(SimState& s) const {
        Kokkos::deep_copy(s.u, 0.0);
        Kokkos::deep_copy(s.v, 0.0);
        Kokkos::deep_copy(s.p, 0.0);
    }
};

struct InflowOutflowIC {
    void apply(SimState& s) const {
        const auto& g = s.grid;
        Kokkos::deep_copy(s.u, 0.0);
        Kokkos::deep_copy(s.v, 0.0);
        Kokkos::deep_copy(s.p, 0.0);
        double u_left = 1.0;
        int i0 = g.u_i_begin();
        auto u = s.u;
        Kokkos::parallel_for("set inflow u",
        Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
        KOKKOS_LAMBDA(const int j){
            u(i0,j) = u_left;
        });

        Kokkos::fence();
    }
};

// Uniform stream u = u_inf, v = 0 everywhere — matches an InflowOutflowBC
// inflow at startup so there is no transient front to flush through the
// domain.
struct UniformStreamIC {
    double u_inf = 1.0;

    UniformStreamIC() = default;
    UniformStreamIC(double u_in) : u_inf(u_in) {}

    void apply(SimState& s) const {
        Kokkos::deep_copy(s.u, u_inf);
        Kokkos::deep_copy(s.v, 0.0);
        Kokkos::deep_copy(s.p, 0.0);
    }
};

struct TaylorGreenIC {
    void apply(SimState& s) const {
        const auto& g  = s.grid;
        const double dx = g.dx;
        const double dy = g.dy;
        const int ng    = MacGrid2D::ng;

        auto u = s.u;
        Kokkos::parallel_for("TG_u",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {g.u_i_begin(), g.u_j_begin()},
                {g.u_i_end(),   g.u_j_end()}),
            KOKKOS_LAMBDA(int i, int j) {
                double x = (i - ng) * dx;
                double y = (j - ng + 0.5) * dy;
                u(i, j) = Kokkos::sin(x) * Kokkos::cos(y);
            });

        auto v = s.v;
        Kokkos::parallel_for("TG_v",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {g.v_i_begin(), g.v_j_begin()},
                {g.v_i_end(),   g.v_j_end()}),
            KOKKOS_LAMBDA(int i, int j) {
                double x = (i - ng + 0.5) * dx;
                double y = (j - ng) * dy;
                v(i, j) = -Kokkos::cos(x) * Kokkos::sin(y);
            });

        Kokkos::deep_copy(s.p, 0.0);
    }
};

