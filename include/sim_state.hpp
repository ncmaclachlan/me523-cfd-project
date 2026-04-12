#pragma once
#include <Kokkos_Core.hpp>
#include "grid.hpp"

struct SimState {
    using View2D = Kokkos::View<double**>;

    MacGrid2D grid;
    View2D u, v, p;
    View2D u_star, v_star;
    View2D rhs;        // pressure Poisson RHS
    double time = 0.0;
    int    step = 0;

    explicit SimState(const MacGrid2D& g)
        : grid(g),
          u     ("u",      g.u_nx(), g.u_ny()),
          v     ("v",      g.v_nx(), g.v_ny()),
          p     ("p",      g.p_nx(), g.p_ny()),
          u_star("u_star", g.u_nx(), g.u_ny()),
          v_star("v_star", g.v_nx(), g.v_ny()),
          rhs   ("rhs",    g.p_nx(), g.p_ny()) {}
};
