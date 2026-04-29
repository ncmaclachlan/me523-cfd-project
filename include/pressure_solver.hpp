#pragma once
#include "sim_state.hpp"
#include <vector>

struct PressureSolveResult {
    int    iters;
    double final_residual;
};

// Per-side boundary condition for the pressure Poisson problem. Dirichlet
// here means homogeneous (p=0) at the half-cell-offset boundary face,
// implemented as ghost = -interior. Neumann means zero-gradient (ghost =
// interior). Periodic wraps around as before.
enum class PressureBC { Periodic, Neumann, Dirichlet };

struct PressureBCSides {
    PressureBC left   = PressureBC::Periodic;
    PressureBC right  = PressureBC::Periodic;
    PressureBC bottom = PressureBC::Periodic;
    PressureBC top    = PressureBC::Periodic;
};

struct PressureSolver {
    int    max_vcycles = 100;
    double tol        = 1e-8;
    int    nu_pre     = 3;
    int    nu_post    = 3;

    void init(const MacGrid2D& grid);
    void set_bc_sides(const PressureBCSides& sides) { bc_sides_ = sides; }
    PressureSolveResult solve(SimState& s, Kokkos::View<double**> rhs);

    // Level must be public so the helper methods below are visible to NVCC.
    // CUDA restriction: __host__ __device__ lambdas (KOKKOS_LAMBDA) cannot
    // appear inside private or protected member functions.
    struct Level {
        int nx, ny;
        double dx, dy;
        Kokkos::View<double**> phi, f, r;
    };

    void fill_ghosts(Kokkos::View<double**> v, int nx, int ny) const;
    void smooth_rbgs(Level& lev, int sweeps) const;
    void compute_residual(Level& lev) const;
    void restrict_to(const Level& fine, Level& coarse) const;
    void prolongate_add(const Level& coarse, Level& fine) const;
    void vcycle(int l) const;
    void subtract_mean(Kokkos::View<double**> v, int nx, int ny) const;

    bool has_dirichlet() const {
        return bc_sides_.left   == PressureBC::Dirichlet
            || bc_sides_.right  == PressureBC::Dirichlet
            || bc_sides_.bottom == PressureBC::Dirichlet
            || bc_sides_.top    == PressureBC::Dirichlet;
    }

private:
    std::vector<Level> levels_;
    double lx_, ly_;
    bool initialized_ = false;
    PressureBCSides bc_sides_{};   // default: all-periodic
};
