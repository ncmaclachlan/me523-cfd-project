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
