#pragma once
#include "sim_state.hpp"
#include <vector>

struct PressureSolveResult {
    int    iters;
    double final_residual;
};

struct PressureSolver {
    int    max_vcycles = 100;
    double tol        = 1e-8;
    int    nu_pre     = 3;
    int    nu_post    = 3;

    void init(const MacGrid2D& grid);
    PressureSolveResult solve(SimState& s, Kokkos::View<double**> rhs);

    // Level must be public so the helper methods below are visible to NVCC.
    // CUDA restriction: __host__ __device__ lambdas (KOKKOS_LAMBDA) cannot
    // appear inside private or protected member functions.
    struct Level {
        int nx, ny;
        double dx, dy;
        Kokkos::View<double**> phi, f, r;
    };

    void periodic_fill(Kokkos::View<double**> v, int nx, int ny) const;
    void smooth_rbgs(Level& lev, int sweeps) const;
    void compute_residual(Level& lev) const;
    void restrict_to(const Level& fine, Level& coarse) const;
    void prolongate_add(const Level& coarse, Level& fine) const;
    void vcycle(int l) const;
    void subtract_mean(Kokkos::View<double**> v, int nx, int ny) const;

private:
    std::vector<Level> levels_;
    double lx_, ly_;
    bool initialized_ = false;
};
