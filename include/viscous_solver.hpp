#pragma once
#include "sim_state.hpp"
#include "physics.hpp"
#include <cmath>

// Solves (I - dt/(2*re) * nabla^2) field = rhs using Red-Black Gauss-Seidel.
// BC::apply_u / apply_v are called at the top of each iteration to keep ghost
// cells consistent as the interior evolves.

struct ViscousSolver {
    int    max_iters = 50;
    double tol       = 1e-6;

    template<typename BC>
    void solve_u(const SimState& s, const BC& bc,
                 Kokkos::View<double**> u_star,
                 Kokkos::View<double**> rhs,
                 double re, double dt) const {
        const double inv_dx2  = 1.0 / (s.grid.dx * s.grid.dx);
        const double inv_dy2  = 1.0 / (s.grid.dy * s.grid.dy);
        const double alpha    = dt / (2.0 * re);
        const double diag_inv = 1.0 / (1.0 + 2.0 * alpha * (inv_dx2 + inv_dy2));
        const int    ncells   = (s.grid.u_i_end() - s.grid.u_i_begin())
                              * (s.grid.u_j_end() - s.grid.u_j_begin());

        for (int iter = 0; iter < max_iters; ++iter) {
            bc.apply_u(s.grid, u_star);

            Kokkos::parallel_for("vs_u_red",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {s.grid.u_i_begin(), s.grid.u_j_begin()},
                    {s.grid.u_i_end(),   s.grid.u_j_end()}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 0) return;
                    u_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(u_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * u_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });

            Kokkos::parallel_for("vs_u_black",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {s.grid.u_i_begin(), s.grid.u_j_begin()},
                    {s.grid.u_i_end(),   s.grid.u_j_end()}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 1) return;
                    u_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(u_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * u_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });

            double res2 = 0.0;
            Kokkos::parallel_reduce("vs_u_res",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {s.grid.u_i_begin(), s.grid.u_j_begin()},
                    {s.grid.u_i_end(),   s.grid.u_j_end()}),
                KOKKOS_LAMBDA(int i, int j, double& lsum) {
                    const double r = u_star(i, j)
                        - alpha * physics::laplacian(u_star, i, j, inv_dx2, inv_dy2)
                        - rhs(i, j);
                    lsum += r * r;
                }, res2);

            if (std::sqrt(res2 / ncells) < tol) break;
        }
    }

    template<typename BC>
    void solve_v(const SimState& s, const BC& bc,
                 Kokkos::View<double**> v_star,
                 Kokkos::View<double**> rhs,
                 double re, double dt) const {
        const double inv_dx2  = 1.0 / (s.grid.dx * s.grid.dx);
        const double inv_dy2  = 1.0 / (s.grid.dy * s.grid.dy);
        const double alpha    = dt / (2.0 * re);
        const double diag_inv = 1.0 / (1.0 + 2.0 * alpha * (inv_dx2 + inv_dy2));
        const int    ncells   = (s.grid.v_i_end() - s.grid.v_i_begin())
                              * (s.grid.v_j_end() - s.grid.v_j_begin());

        for (int iter = 0; iter < max_iters; ++iter) {
            bc.apply_v(s.grid, v_star);

            Kokkos::parallel_for("vs_v_red",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {s.grid.v_i_begin(), s.grid.v_j_begin()},
                    {s.grid.v_i_end(),   s.grid.v_j_end()}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 0) return;
                    v_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(v_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * v_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });

            Kokkos::parallel_for("vs_v_black",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {s.grid.v_i_begin(), s.grid.v_j_begin()},
                    {s.grid.v_i_end(),   s.grid.v_j_end()}),
                KOKKOS_LAMBDA(int i, int j) {
                    if ((i + j) % 2 != 1) return;
                    v_star(i, j) = (rhs(i, j)
                        + alpha * (physics::laplacian(v_star, i, j, inv_dx2, inv_dy2)
                                   + 2.0 * v_star(i, j) * (inv_dx2 + inv_dy2))) * diag_inv;
                });

            double res2 = 0.0;
            Kokkos::parallel_reduce("vs_v_res",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {s.grid.v_i_begin(), s.grid.v_j_begin()},
                    {s.grid.v_i_end(),   s.grid.v_j_end()}),
                KOKKOS_LAMBDA(int i, int j, double& lsum) {
                    const double r = v_star(i, j)
                        - alpha * physics::laplacian(v_star, i, j, inv_dx2, inv_dy2)
                        - rhs(i, j);
                    lsum += r * r;
                }, res2);

            if (std::sqrt(res2 / ncells) < tol) break;
        }
    }
};
