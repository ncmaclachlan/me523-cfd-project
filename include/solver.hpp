#pragma once
#include <cmath>
#include "run_config.hpp"
#include "sim_state.hpp"
#include "boundary_conditions.hpp"
#include "initial_conditions.hpp"
#include "integrator.hpp"
#include "physics.hpp"
#include "pressure_solver.hpp"
#include "output.hpp"

template<typename BC = LidDrivenCavityBC,
         typename Integrator = ForwardEuler>
struct Solver {
    RunConfig      config;
    SimState       state;
    BC             bc;
    Integrator     integrator;
    PressureSolver pressure;

    // Cross-step scratch: owned by Solver, passed explicitly to each stage
    Kokkos::View<double**> u_star, v_star, rhs;

    explicit Solver(const RunConfig& cfg, BC bc_ = {}, Integrator integ = {})
        : config(cfg),
          state(MacGrid2D(cfg), n_steps_estimate(cfg)),
          bc(std::move(bc_)),
          integrator(std::move(integ)),
          u_star("u_star", state.grid.u_nx_total(), state.grid.u_ny_total()),
          v_star("v_star", state.grid.v_nx_total(), state.grid.v_ny_total()),
          rhs   ("rhs",    state.grid.p_nx_total(), state.grid.p_ny_total())
    {
        TaylorGreenIC{}.apply(state);
        pressure.init(state.grid);
    }

    // Conservative upper bound on step count for pre-allocating history arrays.
    // With CFL stepping: assumes u_max <= 2; with fixed dt: uses config.dt.
    static int n_steps_estimate(const RunConfig& cfg) {
        MacGrid2D g(cfg);
        double dt_min;
        if (cfg.cfl > 0.0)
            dt_min = cfg.cfl * std::min(g.dx, g.dy) / 2.0;
        else
            dt_min = cfg.dt;
        return static_cast<int>(std::ceil(cfg.t_end / dt_min)) + 10;
    }

    void advance() {
        // Compute time step: adaptive CFL or fixed
        const double dt = (config.cfl > 0.0)
                          ? physics::compute_cfl_dt(state, config.cfl)
                          : config.dt;

        // 1. Apply boundary conditions
        bc.apply(state);

        // 2. Predict velocity (writes into u_star, v_star)
        integrator.predict(state, bc, u_star, v_star, config.re, dt);

        // 3. Compute pressure Poisson RHS from divergence of u_star
        physics::compute_pressure_rhs(state, u_star, v_star, dt, rhs);

        // 4. Solve pressure Poisson equation (reads rhs, writes s.p)
        pressure.solve(state, rhs);

        // 5. Correct velocity with pressure gradient
        physics::correct_velocity(state, u_star, v_star, dt);

        if (state.step < state.n_steps) {
            state.ke_history(state.step)   = physics::compute_kinetic_energy(state);
            state.div_history(state.step)  = physics::compute_l2_divergence(state);
            state.time_history(state.step) = state.time + dt;
            if (config.diagnostics) {
                auto enorms = physics::compute_error_norms(state, config.re);
                state.err_l2_history(state.step)   = enorms.l2;
                state.err_linf_history(state.step) = enorms.linf;
            }
        }

        state.time += dt;
        ++state.step;
    }

    void run() {
        CSVOutput output(config.output_path());

        while (state.step < state.n_steps) {
            advance();

            /* if (config.output_interval > 0 &&
                state.step % config.output_interval == 0) {
                output.write(state);
            } */
        }

        output.write(state);
        output.write_kinetic_energy(state);
        output.write_l2_divergence(state);
        if (config.diagnostics) {
            output.write_exact(state, config.re);
            output.write_error_norms(state);
        }
    }
};
