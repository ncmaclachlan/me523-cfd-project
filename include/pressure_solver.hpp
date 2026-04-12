#pragma once
#include "sim_state.hpp"

struct PressureSolver {
    void solve(SimState& s, Kokkos::View<double**> rhs) {
        // TODO: solve the pressure Poisson equation
        //   Laplacian(p) = rhs  →  writes result into s.p
        // using Trilinos (Tpetra + Belos/Ifpack2 or AztecOO)
    }
};
