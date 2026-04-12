#pragma once
#include <Kokkos_Core.hpp>
#include "grid.hpp"

struct SimState {
    using View2D = Kokkos::View<double**>;

    MacGrid2D grid;
    View2D u, v, p;
    double time = 0.0;
    int    step = 0;

    explicit SimState(const MacGrid2D& g)
        : grid(g),
          u("u", g.u_nx_total(), g.u_ny_total()),
          v("v", g.v_nx_total(), g.v_ny_total()),
          p("p", g.p_nx_total(), g.p_ny_total()) {}
};
