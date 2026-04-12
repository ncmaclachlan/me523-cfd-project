#pragma once
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

    explicit Solver(const RunConfig& cfg, BC bc_ = {}, Integrator integ = {})
        : config(cfg),
          state(MacGrid2D(cfg)),
          bc(std::move(bc_)),
          integrator(std::move(integ))
    {
        ZeroIC{}.apply(state);
    }

    void advance() {
        // 1. Apply boundary conditions
        bc.apply(state);

        // 2. Predict velocity (u_star, v_star)
        integrator.predict(state, config.re, config.dt);

        // 3. Compute pressure Poisson RHS from divergence of u_star
        physics::compute_pressure_rhs(state, config.dt, state.rhs);

        // 4. Solve pressure Poisson equation
        pressure.solve(state);

        // 5. Correct velocity with pressure gradient
        physics::correct_velocity(state, config.dt);

        state.time += config.dt;
        ++state.step;
    }

    void run() {
        CSVOutput output(config.output_filename);

        while (state.time < config.t_end) {
            advance();

            if (config.output_interval > 0 &&
                state.step % config.output_interval == 0) {
                output.write(state);
            }
        }

        output.write(state);
    }
};
