#pragma once
#include "sim_state.hpp"
#include "physics.hpp"

struct ForwardEuler {
    void predict(const SimState& s,
                 Kokkos::View<double**> u_star,
                 Kokkos::View<double**> v_star,
                 double re, double dt) const {
        // TODO: u_star = u + dt * RHS_u(u, v)
        //       v_star = v + dt * RHS_v(u, v)
        // 1. compute momentum RHS into temporary views
        // 2. u_star(i,j) = u(i,j) + dt * rhs_u(i,j)
        //    v_star(i,j) = v(i,j) + dt * rhs_v(i,j)
    }
};

struct RK2 {
    // Internal scratch for the intermediate stage — allocated on first use
    mutable Kokkos::View<double**> u_tilde, v_tilde;

    void predict(const SimState& s,
                 Kokkos::View<double**> u_star,
                 Kokkos::View<double**> v_star,
                 double re, double dt) const {
        if (u_tilde.extent(0) == 0) {
            u_tilde = Kokkos::View<double**>("u_tilde", s.grid.u_nx_total(), s.grid.u_ny_total());
            v_tilde = Kokkos::View<double**>("v_tilde", s.grid.v_nx_total(), s.grid.v_ny_total());
        }
        // TODO: Heun's method (predictor-corrector)
        // 1. k1_u = RHS_u(u, v);             k1_v = RHS_v(u, v)
        // 2. u_tilde = u + dt*k1_u;          v_tilde = v + dt*k1_v
        // 3. k2_u = RHS_u(u_tilde, v_tilde); k2_v = RHS_v(u_tilde, v_tilde)
        // 4. u_star = u + dt/2*(k1_u + k2_u)
        //    v_star = v + dt/2*(k1_v + k2_v)
    }
};
