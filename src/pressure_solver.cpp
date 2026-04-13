#include "pressure_solver.hpp"
#include "multigrid.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool is_power_of_2(int n) { return n > 0 && (n & (n - 1)) == 0; }

static int ilog2(int n) {
    int k = 0;
    while (n > 1) { n >>= 1; ++k; }
    return k;
}

// ---------------------------------------------------------------------------
// init — build the multigrid level hierarchy
// ---------------------------------------------------------------------------

void PressureSolver::init(const MacGrid2D& grid) {
    if (initialized_) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    lx_ = grid.lx;
    ly_ = grid.ly;

    if (!is_power_of_2(nx) || !is_power_of_2(ny))
        throw std::invalid_argument(
            "PressureSolver: nx and ny must be powers of 2 for geometric multigrid");

    // Coarsest level has 2x2 cells (don't go to 1x1 — degenerate for periodic)
    const int num_levels = ilog2(std::min(nx, ny));  // e.g. 64 -> 6 levels (64..2)

    levels_.resize(num_levels);
    for (int l = 0; l < num_levels; ++l) {
        auto& lev = levels_[l];
        lev.nx = nx >> l;
        lev.ny = ny >> l;
        lev.dx = lx_ / lev.nx;
        lev.dy = ly_ / lev.ny;

        const std::string tag = "mg_l" + std::to_string(l) + "_";
        lev.phi = Kokkos::View<double**>(tag + "phi", lev.nx + 2, lev.ny + 2);
        lev.f   = Kokkos::View<double**>(tag + "f",   lev.nx + 2, lev.ny + 2);
        lev.r   = Kokkos::View<double**>(tag + "r",   lev.nx + 2, lev.ny + 2);
    }

    initialized_ = true;
}

// ---------------------------------------------------------------------------
// periodic_fill — ghost-cell exchange for cell-centered periodic field
// ---------------------------------------------------------------------------

void PressureSolver::periodic_fill(Kokkos::View<double**> v,
                                   int nx, int ny) const {
    // x-direction ghosts
    Kokkos::parallel_for("mg_pfill_x", Kokkos::RangePolicy<>(1, ny + 1),
        KOKKOS_LAMBDA(int j) {
            v(0,      j) = v(nx, j);
            v(nx + 1, j) = v(1,  j);
        });

    // y-direction ghosts
    Kokkos::parallel_for("mg_pfill_y", Kokkos::RangePolicy<>(1, nx + 1),
        KOKKOS_LAMBDA(int i) {
            v(i, 0)      = v(i, ny);
            v(i, ny + 1) = v(i, 1);
        });

    // corners
    Kokkos::parallel_for("mg_pfill_c", Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(int) {
            v(0,      0)      = v(nx, ny);
            v(nx + 1, 0)      = v(1,  ny);
            v(0,      ny + 1) = v(nx, 1);
            v(nx + 1, ny + 1) = v(1,  1);
        });
}

// ---------------------------------------------------------------------------
// smooth_rbgs — red-black Gauss-Seidel
// ---------------------------------------------------------------------------

void PressureSolver::smooth_rbgs(Level& lev, int sweeps) const {
    const int nx = lev.nx;
    const int ny = lev.ny;
    const double inv_dx2 = 1.0 / (lev.dx * lev.dx);
    const double inv_dy2 = 1.0 / (lev.dy * lev.dy);
    const double diag_inv = 1.0 / (2.0 * inv_dx2 + 2.0 * inv_dy2);

    auto phi = lev.phi;
    auto f   = lev.f;

    for (int s = 0; s < sweeps; ++s) {
        // --- Red pass: (i+j) % 2 == 0 ---
        periodic_fill(phi, nx, ny);
        Kokkos::parallel_for("mg_rbgs_red",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nx + 1, ny + 1}),
            KOKKOS_LAMBDA(int i, int j) {
                if ((i + j) % 2 != 0) return;
                phi(i, j) = ((phi(i+1,j) + phi(i-1,j)) * inv_dx2
                           + (phi(i,j+1) + phi(i,j-1)) * inv_dy2
                           - f(i, j)) * diag_inv;
            });

        // --- Black pass: (i+j) % 2 == 1 ---
        periodic_fill(phi, nx, ny);
        Kokkos::parallel_for("mg_rbgs_black",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nx + 1, ny + 1}),
            KOKKOS_LAMBDA(int i, int j) {
                if ((i + j) % 2 != 1) return;
                phi(i, j) = ((phi(i+1,j) + phi(i-1,j)) * inv_dx2
                           + (phi(i,j+1) + phi(i,j-1)) * inv_dy2
                           - f(i, j)) * diag_inv;
            });
    }
}

// ---------------------------------------------------------------------------
// compute_residual — r = f - Lap(phi)
// ---------------------------------------------------------------------------

void PressureSolver::compute_residual(Level& lev) const {
    const int nx = lev.nx;
    const int ny = lev.ny;
    const double inv_dx2 = 1.0 / (lev.dx * lev.dx);
    const double inv_dy2 = 1.0 / (lev.dy * lev.dy);

    periodic_fill(lev.phi, nx, ny);

    auto phi = lev.phi;
    auto f   = lev.f;
    auto r   = lev.r;

    Kokkos::parallel_for("mg_residual",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nx + 1, ny + 1}),
        KOKKOS_LAMBDA(int i, int j) {
            const double lap = (phi(i+1,j) - 2.0*phi(i,j) + phi(i-1,j)) * inv_dx2
                             + (phi(i,j+1) - 2.0*phi(i,j) + phi(i,j-1)) * inv_dy2;
            r(i, j) = f(i, j) - lap;
        });
}

// ---------------------------------------------------------------------------
// restrict_to — full-weighting restriction (cell-centered 2x2 average)
// ---------------------------------------------------------------------------

void PressureSolver::restrict_to(const Level& fine, Level& coarse) const {
    periodic_fill(const_cast<Level&>(fine).r, fine.nx, fine.ny);

    const int nc_x = coarse.nx;
    const int nc_y = coarse.ny;
    auto r_f = fine.r;
    auto f_c = coarse.f;

    Kokkos::parallel_for("mg_restrict",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nc_x + 1, nc_y + 1}),
        KOKKOS_LAMBDA(int ic, int jc) {
            // Coarse cell (ic,jc) covers fine cells (2*ic-1,2*jc-1)..(2*ic,2*jc)
            const int if0 = 2 * ic - 1;
            const int jf0 = 2 * jc - 1;
            f_c(ic, jc) = 0.25 * (r_f(if0,   jf0)   + r_f(if0+1, jf0)
                                 + r_f(if0,   jf0+1) + r_f(if0+1, jf0+1));
        });
}

// ---------------------------------------------------------------------------
// prolongate_add — bilinear interpolation (cell-centered, 3/4 + 1/4 weights)
// ---------------------------------------------------------------------------

void PressureSolver::prolongate_add(const Level& coarse, Level& fine) const {
    periodic_fill(const_cast<Level&>(coarse).phi, coarse.nx, coarse.ny);

    const int nf_x = fine.nx;
    const int nf_y = fine.ny;
    auto phi_c = coarse.phi;
    auto phi_f = fine.phi;

    Kokkos::parallel_for("mg_prolongate",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nf_x + 1, nf_y + 1}),
        KOKKOS_LAMBDA(int i_f, int j_f) {
            // Determine which coarse cell owns this fine cell
            const int ic = (i_f - 1) / 2 + 1;
            const int jc = (j_f - 1) / 2 + 1;

            // Sub-cell position: 0 = near side, 1 = far side
            const int si = (i_f - 1) % 2;
            const int sj = (j_f - 1) % 2;

            // Neighbor direction: left/down (si==0) or right/up (si==1)
            const int di = (si == 0) ? -1 : 1;
            const int dj = (sj == 0) ? -1 : 1;

            // Bilinear weights (tensor product of 1D: 3/4 near, 1/4 far)
            phi_f(i_f, j_f) += 0.5625 * phi_c(ic,      jc)        // 9/16
                             + 0.1875 * phi_c(ic + di,  jc)        // 3/16
                             + 0.1875 * phi_c(ic,       jc + dj)   // 3/16
                             + 0.0625 * phi_c(ic + di,  jc + dj);  // 1/16
        });
}

// ---------------------------------------------------------------------------
// subtract_mean — enforce zero-mean gauge
// ---------------------------------------------------------------------------

void PressureSolver::subtract_mean(Kokkos::View<double**> v,
                                   int nx, int ny) const {
    double sum = 0.0;
    Kokkos::parallel_reduce("mg_mean",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nx + 1, ny + 1}),
        KOKKOS_LAMBDA(int i, int j, double& lsum) {
            lsum += v(i, j);
        }, sum);

    const double mean = sum / static_cast<double>(nx * ny);

    Kokkos::parallel_for("mg_sub_mean",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nx + 1, ny + 1}),
        KOKKOS_LAMBDA(int i, int j) {
            v(i, j) -= mean;
        });
}

// ---------------------------------------------------------------------------
// vcycle — recursive V-cycle
// ---------------------------------------------------------------------------

void PressureSolver::vcycle(int l) const {
    const int nlev = static_cast<int>(levels_.size());

    if (l == nlev - 1) {
        // Coarsest level: iterate many sweeps (cheap — only 2x2 = 4 cells)
        const_cast<PressureSolver*>(this)->smooth_rbgs(
            const_cast<Level&>(levels_[l]), 50);
        return;
    }

    auto& fine   = const_cast<Level&>(levels_[l]);
    auto& coarse = const_cast<Level&>(levels_[l + 1]);

    // Pre-smooth
    const_cast<PressureSolver*>(this)->smooth_rbgs(fine, nu_pre);

    // Compute residual
    const_cast<PressureSolver*>(this)->compute_residual(fine);

    // Restrict residual to coarse RHS
    const_cast<PressureSolver*>(this)->restrict_to(fine, coarse);

    // Zero coarse-grid correction
    Kokkos::deep_copy(coarse.phi, 0.0);

    // Recurse
    vcycle(l + 1);

    // Prolongate correction and add to fine solution
    const_cast<PressureSolver*>(this)->prolongate_add(coarse, fine);

    // Post-smooth
    const_cast<PressureSolver*>(this)->smooth_rbgs(fine, nu_post);
}

// ---------------------------------------------------------------------------
// solve — outer iteration loop
// ---------------------------------------------------------------------------

PressureSolveResult PressureSolver::solve(SimState& s,
                                          Kokkos::View<double**> rhs) {
    if (!initialized_) init(s.grid);

    const int nx = levels_[0].nx;
    const int ny = levels_[0].ny;

    // Copy RHS and initial guess into level 0
    Kokkos::deep_copy(levels_[0].f,   rhs);
    Kokkos::deep_copy(levels_[0].phi, s.p);

    // Enforce compatibility: mean(f) = 0
    subtract_mean(levels_[0].f, nx, ny);

    double rnorm = 0.0;

    for (int iter = 0; iter < max_vcycles; ++iter) {
        vcycle(0);

        // Enforce mean-zero gauge on solution
        subtract_mean(levels_[0].phi, nx, ny);

        // Check convergence
        compute_residual(levels_[0]);

        rnorm = 0.0;
        auto r = levels_[0].r;
        Kokkos::parallel_reduce("mg_rnorm",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {nx + 1, ny + 1}),
            KOKKOS_LAMBDA(int i, int j, double& lsum) {
                lsum += r(i, j) * r(i, j);
            }, rnorm);
        rnorm = std::sqrt(rnorm / static_cast<double>(nx * ny));

        if (rnorm < tol) {
            periodic_fill(levels_[0].phi, nx, ny);
            Kokkos::deep_copy(s.p, levels_[0].phi);
            return {iter + 1, rnorm};
        }
    }

    // Hit max iterations — copy solution anyway
    periodic_fill(levels_[0].phi, nx, ny);
    Kokkos::deep_copy(s.p, levels_[0].phi);
    return {max_vcycles, rnorm};
}
