#pragma once
#include <limits>

struct RunStats {
    int    n_steps      = 0;
    double dt_min       = std::numeric_limits<double>::max();
    double dt_max       = 0.0;
    double dt_sum       = 0.0;

    // RBGS viscous solver (u and v combined); only valid when has_rbgs == true
    bool   has_rbgs          = false;
    int    rbgs_iters_min    = 0;
    int    rbgs_iters_max    = 0;
    long   rbgs_iters_total  = 0;
    double rbgs_res_max      = 0.0;

    // Pressure multigrid solver
    int    pres_iters_min    = std::numeric_limits<int>::max();
    int    pres_iters_max    = 0;
    long   pres_iters_total  = 0;
    double pres_res_max      = 0.0;

    // Per-stage accumulated wall times (seconds)
    double wall_bc             = 0.0;
    double wall_predict        = 0.0;
    double wall_pressure_rhs   = 0.0;
    double wall_pressure_solve = 0.0;
    double wall_correct        = 0.0;
    double wall_total          = 0.0;
};
