#pragma once
#include <Kokkos_Core.hpp>

template<typename Traits>
struct SimState {
    using Grid     = typename Traits::Grid;
    using Scalar   = typename Traits::Scalar;
    using MemSpace = typename Traits::ExecSpace::memory_space;
    using View2D   = Kokkos::View<Scalar**, MemSpace>;

    Grid   grid;
    View2D u, v, p;
    View2D u_star, v_star;
    Scalar time = Scalar(0);
    int    step  = 0;

    explicit SimState(Grid g)
        : grid(g),
          u     ("u",      g.u_nx(), g.u_ny()),
          v     ("v",      g.v_nx(), g.v_ny()),
          p     ("p",      g.p_nx(), g.p_ny()),
          u_star("u_star", g.u_nx(), g.u_ny()),
          v_star("v_star", g.v_nx(), g.v_ny()) {}
};
