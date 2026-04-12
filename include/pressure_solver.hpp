#pragma once
#include "sim_state.hpp"

struct PressureSolver {
    void solve(SimState& s) {
        // TODO: solve the pressure Poisson equation
        //   Laplacian(p) = rhs
        // using Trilinos (Tpetra + Belos/Ifpack2 or AztecOO)
    }
};
