#pragma once
#include <Kokkos_Core.hpp>

struct ForwardEuler {
    template<typename State, typename Physics, typename BC, typename Scalar>
    void step(State& s, const Physics& phys, const BC& bc, Scalar dt) const {
        // TODO: u_new = u + dt * RHS(u, p)
    }
};

struct RK2 {
    template<typename State, typename Physics, typename BC, typename Scalar>
    void step(State& s, const Physics& phys, const BC& bc, Scalar dt) const {
        // TODO: Runge-Kutta 2 (Heun) step
    }
};
