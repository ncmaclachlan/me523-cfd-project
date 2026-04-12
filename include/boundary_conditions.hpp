#pragma once
#include "sim_state.hpp"

struct LidDrivenCavityBC {
    double lid_velocity = 1.0;

    LidDrivenCavityBC() = default;
    LidDrivenCavityBC(double u_lid) : lid_velocity(u_lid) {}

    void apply(SimState& s) const {
        // TODO: no-slip on left/right/bottom walls, lid velocity on top boundary
    }
};

struct PeriodicBC {
    void apply(SimState& s) const {
        // TODO: periodic wrapping in x and y
    }
};
