#pragma once
#include <Kokkos_Core.hpp>

struct ZeroIC {
    template<typename State>
    void apply(State& s) const {
        Kokkos::deep_copy(s.u, 0.0);
        Kokkos::deep_copy(s.v, 0.0);
        Kokkos::deep_copy(s.p, 0.0);
    }
};

struct TaylorGreenIC {
    template<typename State>
    void apply(State& s) const {
        // TODO: u = sin(x)cos(y), v = -cos(x)sin(y), p = -(cos(2x)+cos(2y))/4
    }
};
