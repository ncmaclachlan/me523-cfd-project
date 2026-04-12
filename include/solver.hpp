#pragma once
#include "sim_state.hpp"

template<typename Traits>
struct Solver {
    using Scalar     = typename Traits::Scalar;
    using Physics    = typename Traits::Physics;
    using BC         = typename Traits::BC;
    using Integrator = typename Traits::Integrator;

    SimState<Traits> state;
    Physics          physics;
    BC               bc;
    Integrator       integrator;

    explicit Solver(typename Traits::Grid     g,
                    typename Traits::Physics  phys  = {},
                    typename Traits::BC       bc_   = {},
                    typename Traits::Integrator integ = {})
        : state(g), physics(phys), bc(bc_), integrator(integ)
    {
        typename Traits::InitialCondition{}.apply(state);
    }

    void advance(Scalar dt) {
        bc.apply(state);
        integrator.step(state, physics, bc, dt);
        state.time += dt;
        ++state.step;
    }
};
