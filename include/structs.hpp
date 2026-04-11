#pragma once
#include <string>
#include <Kokkos_Core.hpp>

struct RunConfig {
    int    nx = 64;
    int    ny = 64;
    double lx = 1.0;
    double ly = 1.0;
    double dt = 1e-3;
    double t_end = 1.0;
    double re = 100.0;          // Reynolds number
    std::string initial_conds = "zero";
    std::string output_file   = "output.csv";
};

struct SimState {
    // Grid coordinates (size nx*ny, row-major)
    Kokkos::View<double*> x;
    Kokkos::View<double*> y;
    double dx = 0.0;
    double dy = 0.0;

    // Velocity components
    Kokkos::View<double*> u;
    Kokkos::View<double*> v;

    // Pressure
    Kokkos::View<double*> p;

    // Grid dimensions (copied from config for convenience)
    int nx = 0;
    int ny = 0;

    // Current simulation time
    double time = 0.0;
};
