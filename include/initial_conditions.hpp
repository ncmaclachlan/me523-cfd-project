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
        // TODO: u = sin(x)cos(y), v = -cos(x)sin(y), p = -(cos(2x)+cos(2y))/4
    }
};
