#pragma once
#include "sim_state.hpp"
#include "physics.hpp"
#include "viscous_solver.hpp"

struct ForwardEuler {
    mutable Kokkos::View<double**> rhs_u, rhs_v;

    template<typename BC>
    void predict(const SimState& s, const BC& /*bc*/,
                 Kokkos::View<double**> u_star,
                 Kokkos::View<double**> v_star,
                 double re, double dt) const {
        if (rhs_u.extent(0) == 0) {
            rhs_u = Kokkos::View<double**>("fe_rhs_u", s.grid.u_nx_total(), s.grid.u_ny_total());
            rhs_v = Kokkos::View<double**>("fe_rhs_v", s.grid.v_nx_total(), s.grid.v_ny_total());
        }

        physics::compute_u_rhs(s, re, rhs_u);
        physics::compute_v_rhs(s, re, rhs_v);

        SimState::View2D u       = s.u;
        SimState::View2D v       = s.v;
        auto             local_rhs_u = rhs_u;
        auto             local_rhs_v = rhs_v;

        Kokkos::parallel_for("fe_u_star",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {s.grid.u_i_begin(), s.grid.u_j_begin()},
                {s.grid.u_i_end(),   s.grid.u_j_end()}),
            KOKKOS_LAMBDA(int i, int j) {
                u_star(i, j) = u(i, j) + dt * local_rhs_u(i, j);
            });

        Kokkos::parallel_for("fe_v_star",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {s.grid.v_i_begin(), s.grid.v_j_begin()},
                {s.grid.v_i_end(),   s.grid.v_j_end()}),
            KOKKOS_LAMBDA(int i, int j) {
                v_star(i, j) = v(i, j) + dt * local_rhs_v(i, j);
            });
    }
};

struct RK2 {
    mutable Kokkos::View<double**> u_tilde, v_tilde;
    mutable Kokkos::View<double**> rhs_u,   rhs_v;
    mutable Kokkos::View<double**> rhs_u2,  rhs_v2;

    template<typename BC>
    void predict(const SimState& s, const BC& /*bc*/,
                 Kokkos::View<double**> u_star,
                 Kokkos::View<double**> v_star,
                 double re, double dt) const {
        if (u_tilde.extent(0) == 0) {
            u_tilde = Kokkos::View<double**>("u_tilde", s.grid.u_nx_total(), s.grid.u_ny_total());
            v_tilde = Kokkos::View<double**>("v_tilde", s.grid.v_nx_total(), s.grid.v_ny_total());
            rhs_u   = Kokkos::View<double**>("rk2_rhs_u",  s.grid.u_nx_total(), s.grid.u_ny_total());
            rhs_v   = Kokkos::View<double**>("rk2_rhs_v",  s.grid.v_nx_total(), s.grid.v_ny_total());
            rhs_u2  = Kokkos::View<double**>("rk2_rhs_u2", s.grid.u_nx_total(), s.grid.u_ny_total());
            rhs_v2  = Kokkos::View<double**>("rk2_rhs_v2", s.grid.v_nx_total(), s.grid.v_ny_total());
        }

        // TODO: Heun's method — requires BC applied to u_tilde/v_tilde at stage 2
        // k1 = RHS(u, v)
        // u_tilde = u + dt * k1
        // k2 = RHS(u_tilde, v_tilde)   <- needs bc.apply_u/v on tilde fields
        // u_star = u + dt/2 * (k1 + k2)
    }
};

struct CrankThatNicolson {
    mutable Kokkos::View<double**> rhs_u_conv, rhs_v_conv;
    mutable Kokkos::View<double**> rhs_u_diff, rhs_v_diff;
    mutable Kokkos::View<double**> rhs_u,      rhs_v;

    ViscousSolver viscous_solver;

    mutable ViscousSolveResult last_u_result{0, 0.0};
    mutable ViscousSolveResult last_v_result{0, 0.0};

    template<typename BC>
    void predict(const SimState& s, const BC& bc,
                 Kokkos::View<double**> u_star,
                 Kokkos::View<double**> v_star,
                 double re, double dt) const {
        if (rhs_u_conv.extent(0) == 0) {
            rhs_u_conv = Kokkos::View<double**>("rhs_u_conv", s.grid.u_nx_total(), s.grid.u_ny_total());
            rhs_v_conv = Kokkos::View<double**>("rhs_v_conv", s.grid.v_nx_total(), s.grid.v_ny_total());
            rhs_u_diff = Kokkos::View<double**>("rhs_u_diff", s.grid.u_nx_total(), s.grid.u_ny_total());
            rhs_v_diff = Kokkos::View<double**>("rhs_v_diff", s.grid.v_nx_total(), s.grid.v_ny_total());
            rhs_u      = Kokkos::View<double**>("rhs_u",      s.grid.u_nx_total(), s.grid.u_ny_total());
            rhs_v      = Kokkos::View<double**>("rhs_v",      s.grid.v_nx_total(), s.grid.v_ny_total());
        }

        physics::compute_u_conv_rhs(s, rhs_u_conv);
        physics::compute_v_conv_rhs(s, rhs_v_conv);
        physics::compute_u_diff_rhs(s, re, rhs_u_diff);
        physics::compute_v_diff_rhs(s, re, rhs_v_diff);

        // Explicit CN RHS: u + dt*conv + (dt/2)*diff
        SimState::View2D         u                = s.u;
        SimState::View2D         v                = s.v;
        Kokkos::View<double**>   local_rhs_u      = rhs_u;
        Kokkos::View<double**>   local_rhs_v      = rhs_v;
        Kokkos::View<double**>   local_rhs_u_conv = rhs_u_conv;
        Kokkos::View<double**>   local_rhs_v_conv = rhs_v_conv;
        Kokkos::View<double**>   local_rhs_u_diff = rhs_u_diff;
        Kokkos::View<double**>   local_rhs_v_diff = rhs_v_diff;

        Kokkos::parallel_for("cn_rhs_u",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {s.grid.u_i_begin(), s.grid.u_j_begin()},
                {s.grid.u_i_end(),   s.grid.u_j_end()}),
            KOKKOS_LAMBDA(int i, int j) {
                local_rhs_u(i, j) = u(i, j)
                            + dt       * local_rhs_u_conv(i, j)
                            + 0.5 * dt * local_rhs_u_diff(i, j);
            });

        Kokkos::parallel_for("cn_rhs_v",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {s.grid.v_i_begin(), s.grid.v_j_begin()},
                {s.grid.v_i_end(),   s.grid.v_j_end()}),
            KOKKOS_LAMBDA(int i, int j) {
                local_rhs_v(i, j) = v(i, j)
                            + dt       * local_rhs_v_conv(i, j)
                            + 0.5 * dt * local_rhs_v_diff(i, j);
            });

        // Initial guess from current velocity
        Kokkos::deep_copy(u_star, s.u);
        Kokkos::deep_copy(v_star, s.v);

        // Implicit diffusion solve
        last_u_result = viscous_solver.solve_u(s, bc, u_star, rhs_u, re, dt);
        last_v_result = viscous_solver.solve_v(s, bc, v_star, rhs_v, re, dt);
    }
};
