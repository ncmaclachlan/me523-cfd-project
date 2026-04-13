#pragma once
#include "sim_state.hpp"
#include "physics.hpp"

// Kim-Moin integrator: Adams-Bashforth 2 (convective) + Crank-Nicolson (viscous)
// per eq. 88 in prob1.tex. The implicit viscous step is solved via ADI — two
// sweeps of 1D tridiagonal solves. The ADI solver is a private nested struct;
// it is only ever needed here so no separate file is warranted.
// Requires tridiag.hpp (issue #54) for the Thomas / Sherman-Morrison solver.
struct KimMoin {
    SimState::View2D H_u_prev, H_v_prev;
    bool first_step = true;

    struct HelmholtzADI {
        // TODO: preallocate work arrays to avoid allocation inside the sweep loops

        // Solve (I - alpha ∇²) out = rhs via ADI (x-sweep then y-sweep).
        // alpha = dt / (2 * Re). Grid spacing and extents come from s.grid.
        void solve(SimState::View2D out,
                   const SimState::View2D& rhs,
                   double alpha,
                   const SimState& s) {
            // TODO: x-sweep — for each row j, solve tridiagonal system in i
            //       subdiag = -alpha/dx^2, diag = 1 + 2*alpha/dx^2, superdiag = -alpha/dx^2
            //       periodic BCs: use Sherman-Morrison from tridiag.hpp
            // TODO: y-sweep — for each col i, solve tridiagonal system in j
            //       subdiag = -alpha/dy^2, diag = 1 + 2*alpha/dy^2, superdiag = -alpha/dy^2
            //       periodic BCs: use Sherman-Morrison from tridiag.hpp
        }
    };
    HelmholtzADI adi;

    void predict(const SimState& s,
                 SimState::View2D u_star,
                 SimState::View2D v_star,
                 double re, double dt) {
        if (H_u_prev.extent(0) == 0) {
            H_u_prev = SimState::View2D("H_u_prev", s.grid.u_nx_total(), s.grid.u_ny_total());
            H_v_prev = SimState::View2D("H_v_prev", s.grid.v_nx_total(), s.grid.v_ny_total());
        }

        // TODO: 1. Compute convective RHS H_u^n, H_v^n via physics::compute_u_rhs / v_rhs

        // TODO: 2. AB2 extrapolation:
        //             H^{n+1/2} = 1.5 * H^n - 0.5 * H^{n-1}
        //          First step: set H^{n-1} = H^n (bootstraps to forward Euler)

        // TODO: 3. Explicit RHS for Helmholtz solve:
        //             rhs_u = u^n + dt * H_u^{n+1/2} + (dt/(2 Re)) * laplacian(u^n)
        //             rhs_v = v^n + dt * H_v^{n+1/2} + (dt/(2 Re)) * laplacian(v^n)

        // TODO: 4. Solve implicit viscous step via ADI:
        //             adi.solve(u_star, rhs_u, dt / (2.0 * re), s);
        //             adi.solve(v_star, rhs_v, dt / (2.0 * re), s);

        // TODO: 5. H_u_prev = H_u^n; H_v_prev = H_v^n; first_step = false;
    }
};

struct ForwardEuler {
    void predict(const SimState& s,
                 SimState::View2D u_star,
                 SimState::View2D v_star,
                 double re, double dt) {
        // TODO: u_star = u + dt * RHS_u(u, v)
        //       v_star = v + dt * RHS_v(u, v)
        // 1. compute momentum RHS into temporary views
        // 2. u_star(i,j) = u(i,j) + dt * rhs_u(i,j)
        //    v_star(i,j) = v(i,j) + dt * rhs_v(i,j)
    }
};

struct RK2 {
    SimState::View2D u_tilde, v_tilde;

    void predict(const SimState& s,
                 SimState::View2D u_star,
                 SimState::View2D v_star,
                 double re, double dt) {
        if (u_tilde.extent(0) == 0) {
            u_tilde = SimState::View2D("u_tilde", s.grid.u_nx_total(), s.grid.u_ny_total());
            v_tilde = SimState::View2D("v_tilde", s.grid.v_nx_total(), s.grid.v_ny_total());
        }
        // TODO: Heun's method (predictor-corrector)
        // 1. k1_u = RHS_u(u, v);             k1_v = RHS_v(u, v)
        // 2. u_tilde = u + dt*k1_u;          v_tilde = v + dt*k1_v
        // 3. k2_u = RHS_u(u_tilde, v_tilde); k2_v = RHS_v(u_tilde, v_tilde)
        // 4. u_star = u + dt/2*(k1_u + k2_u)
        //    v_star = v + dt/2*(k1_v + k2_v)
    }
};
