#pragma once
#include <Kokkos_Core.hpp>

struct LidDrivenCavityBC {
    double lid_velocity = 1.0;

    explicit LidDrivenCavityBC(double u_lid = 1.0) : lid_velocity(u_lid) {}

    template<typename State>
    void apply(State& s) const {
        // TODO: no-slip on left/right/bottom walls, lid velocity on top boundary
    }
};

struct PeriodicBC {
    template<typename State>
    void apply(State& s) const {
        // TODO: periodic wrapping in x and y
    }
};
