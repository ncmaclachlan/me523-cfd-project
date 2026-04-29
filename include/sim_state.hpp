#pragma once
#include <Kokkos_Core.hpp>
#include "grid.hpp"

struct SimState2D {
    using View2D     = Kokkos::View<double**>;
    using View1D     = Kokkos::View<double*>;
    using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;

    MacGrid2D grid;
    View2D u, v, p, div;
    HostView1D ke_history, div_history, time_history,
               err_l2_history, err_linf_history;
    int    n_steps;
    double time = 0.0;
    int    step = 0;

    SimState(const MacGrid2D& g, int n_steps_)
        : grid(g),
          u("u", g.u_nx_total(), g.u_ny_total()),
          v("v", g.v_nx_total(), g.v_ny_total()),
          p("p", g.p_nx_total(), g.p_ny_total()),
          div("div", g.p_nx_total(), g.p_ny_total()),
          ke_history("ke_history", n_steps_),
          div_history("div_history", n_steps_),
          time_history("time_history", n_steps_),
          err_l2_history("err_l2_history", n_steps_),
          err_linf_history("err_linf_history", n_steps_),
          n_steps(n_steps_) {}
};

struct SimState3D {
    using View3D     = Kokkos::View<double***>;
    using View1D     = Kokkos::View<double*>;
    using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;

    MacGrid3D grid;
    View3D u, v, w, p, div;
    HostView1D ke_history, div_history, time_history,
               err_l2_history, err_linf_history;
    int    n_steps;
    double time = 0.0;
    int    step = 0;

    SimState(const MacGrid3D& g, int n_steps_)
        : grid(g),
          u("u", g.u_nx_total(), g.u_ny_total(), g.u_nz_total()),
          v("v", g.v_nx_total(), g.v_ny_total(), g.v_nz_total()),
          w("w", g.w_nx_total(), g.w_ny_total(), g.w_nz_total()),
          p("p", g.p_nx_total(), g.p_ny_total(), g.p_nz_total()),
          div("div", g.p_nx_total(), g.p_ny_total(), g.p_nz_total()),
          ke_history("ke_history", n_steps_),
          div_history("div_history", n_steps_),
          time_history("time_history", n_steps_),
          err_l2_history("err_l2_history", n_steps_),
          err_linf_history("err_linf_history", n_steps_),
          n_steps(n_steps_) {}
};