#pragma once
#include "sim_state.hpp"
#include "physics.hpp"

struct ForwardEuler {
    void predict(SimState& s, double re, double dt) const {
        // TODO: u_star = u + dt * RHS_u(u, v)
        //       v_star = v + dt * RHS_v(u, v)
        // 1. compute momentum RHS into temporary views
        // 2. u_star(i,j) = u(i,j) + dt * rhs_u(i,j)
        //    v_star(i,j) = v(i,j) + dt * rhs_v(i,j)
    }
};

struct RK2 {
    void predict(SimState& s, double re, double dt) const {
        // TODO: Heun's method (predictor-corrector)
        // 1. k1_u = RHS_u(u, v);             k1_v = RHS_v(u, v)
        // 2. u_tilde = u + dt*k1_u;          v_tilde = v + dt*k1_v
        // 3. k2_u = RHS_u(u_tilde, v_tilde); k2_v = RHS_v(u_tilde, v_tilde)
        // 4. u_star = u + dt/2*(k1_u + k2_u)
        //    v_star = v + dt/2*(k1_v + k2_v)
    }
};
