#pragma once
#include <Kokkos_Core.hpp>
#include "grid.hpp"

class SimState {
public:
    // builds SimState from a grid object
    explicit SimState(const Grid& grid);

    const Grid& grid() const { return grid_; }
    // allows future adaptive mesh refinement, but for now do not allow grid
    // modifications
    // Grid& grid() { return grid_; }

    Kokkos::View<double**> u;
    Kokkos::View<double**> v;
    Kokkos::View<double**> p;

    Kokkos::View<double**> u_star;
    Kokkos::View<double**> v_star;

    double time = 0.0;
    int step = 0;

private:
    Grid grid_;
};