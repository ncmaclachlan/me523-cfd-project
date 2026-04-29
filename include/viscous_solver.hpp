#pragma once
#include "sim_state.hpp"
#include "boundary_conditions.hpp"
#include "physics.hpp"
#include <cmath>

// Solves (I - dt/(2*re) * nabla^2) field = rhs using Red-Black Gauss-Seidel.
// BC::apply_u / apply_v are called at the top of each iteration to keep ghost
// cells consistent as the interior evolves.

struct ViscousSolveResult {
    int    iters;
    double final_residual;
};

struct ViscousSolver {
    int    max_iters = 500;
    double tol       = 1e-8;

    template<typename BC>
    ViscousSolveResult solve_u(const SimState& s, const BC& bc,
                               Kokkos::View<double**> u_star,
                               Kokkos::View<double**> rhs,
                               double re, double dt) const {
        const double inv_dx2  = 1.0 / (s.grid.dx * s.grid.dx);
        const double inv_dy2  = 1.0 / (s.grid.dy * s.grid.dy);
        const double alpha    = dt / (2.0 * re);
        const double diag_inv = 1.0 / (1.0 + 2.0 * alpha * (inv_dx2 + inv_dy2));

        const UVRange r_u  = BC::u_solve_range(s.grid);
        const int    ncells = (r_u.i_end - r_u.i_begin) * (r_u.j_end - r_u.j_begin);
        if (ncells <= 0) return {0, 0.0};

        double rms = tol + 1.0;
        int    iter = 0;
        for (; iter < max_iters; ++iter) {
            bc.apply_u(s.grid, u_star);

            Kokkos::parallel_for("vs_u_red",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {r_u.i_begin, r_u.j_begin},
                    {r_u.i_end,   r_u.j_end}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 0) return;
                    u_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(u_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * u_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });
            Kokkos::fence();

            bc.apply_u(s.grid, u_star);

            Kokkos::parallel_for("vs_u_black",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {r_u.i_begin, r_u.j_begin},
                    {r_u.i_end,   r_u.j_end}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 1) return;
                    u_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(u_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * u_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });

            Kokkos::fence();

            bc.apply_u(s.grid, u_star);

            double res2 = 0.0;
            Kokkos::parallel_reduce("vs_u_res",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {r_u.i_begin, r_u.j_begin},
                    {r_u.i_end,   r_u.j_end}),
                KOKKOS_LAMBDA(int i, int j, double& lsum) {
                    const double r = u_star(i, j)
                        - alpha * physics::laplacian(u_star, i, j, inv_dx2, inv_dy2)
                        - rhs(i, j);
                    lsum += r * r;
                }, res2);
            Kokkos::fence();
            rms = std::sqrt(res2 / ncells);
            if (rms < tol) { ++iter; break; }
        }
        return {iter, rms};
    }

    template<typename BC>
    ViscousSolveResult solve_v(const SimState& s, const BC& bc,
                               Kokkos::View<double**> v_star,
                               Kokkos::View<double**> rhs,
                               double re, double dt) const {
        const double inv_dx2  = 1.0 / (s.grid.dx * s.grid.dx);
        const double inv_dy2  = 1.0 / (s.grid.dy * s.grid.dy);
        const double alpha    = dt / (2.0 * re);
        const double diag_inv = 1.0 / (1.0 + 2.0 * alpha * (inv_dx2 + inv_dy2));

        const UVRange r_v  = BC::v_solve_range(s.grid);
        const int    ncells = (r_v.i_end - r_v.i_begin) * (r_v.j_end - r_v.j_begin);
        if (ncells <= 0) return {0, 0.0};

        double rms = tol + 1.0;
        int    iter = 0;
        for (; iter < max_iters; ++iter) {
            bc.apply_v(s.grid, v_star);

            Kokkos::parallel_for("vs_v_red",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {r_v.i_begin, r_v.j_begin},
                    {r_v.i_end,   r_v.j_end}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 0) return;
                    v_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(v_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * v_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });

            Kokkos::fence();

            bc.apply_v(s.grid, v_star);

            Kokkos::parallel_for("vs_v_black",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {r_v.i_begin, r_v.j_begin},
                    {r_v.i_end,   r_v.j_end}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 1) return;
                    v_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(v_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * v_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });

            Kokkos::fence();

            bc.apply_v(s.grid, v_star);

            double res2 = 0.0;
            Kokkos::parallel_reduce("vs_v_res",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {r_v.i_begin, r_v.j_begin},
                    {r_v.i_end,   r_v.j_end}),
                KOKKOS_LAMBDA(int i, int j, double& lsum) {
                    const double r = v_star(i, j)
                        - alpha * physics::laplacian(v_star, i, j, inv_dx2, inv_dy2)
                        - rhs(i, j);
                    lsum += r * r;
                }, res2);
            Kokkos::fence();

            rms = std::sqrt(res2 / ncells);
            if (rms < tol) { ++iter; break; }
        }
        return {iter, rms};
    }
};
