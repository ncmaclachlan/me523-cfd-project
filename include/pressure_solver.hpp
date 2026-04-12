#pragma once
#include "sim_state.hpp"

struct PressureSolver {
    void solve(SimState& s, Kokkos::View<double**> rhs);
};
