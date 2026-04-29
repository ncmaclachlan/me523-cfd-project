#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>
#include "run_config.hpp"
#include "run_stats.hpp"
#include "sim_state.hpp"
#include "boundary_conditions.hpp"
#include "initial_conditions.hpp"
#include "integrator.hpp"
#include "physics.hpp"
#include "pressure_solver.hpp"
#include "output.hpp"

template<typename BC = PeriodicBC,
         typename Integrator = CrankThatNicolson,
         typename IC = TaylorGreenIC>
struct Solver {
    RunConfig      config;
    SimState       state;
    BC             bc;
    Integrator     integrator;
    IC             ic;
    PressureSolver pressure;

    // Cross-step scratch: owned by Solver, passed explicitly to each stage
    Kokkos::View<double**> u_star, v_star, rhs;

    RunStats stats_;

    explicit Solver(const RunConfig& cfg, BC bc_ = {}, Integrator integ = {}, IC ic_ = {})
        : config(cfg),
          state(MacGrid2D(cfg), n_steps_estimate(cfg)),
          bc(std::move(bc_)),
          integrator(std::move(integ)),
          ic(std::move(ic_)),
          u_star("u_star", state.grid.u_nx_total(), state.grid.u_ny_total()),
          v_star("v_star", state.grid.v_nx_total(), state.grid.v_ny_total()),
          rhs   ("rhs",    state.grid.p_nx_total(), state.grid.p_ny_total())
    {
        ic.apply(state);
        pressure.init(state.grid);
        pressure.set_bc_sides(BC::pressure_sides());
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
        using Clock = std::chrono::steady_clock;
        auto dur = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };

        const double dt = (config.cfl > 0.0)
                          ? physics::compute_cfl_dt(state, config.cfl)
                          : config.dt;

        // 1. Apply boundary conditions
        Kokkos::fence(); auto t0 = Clock::now();
        bc.apply(state);

        // 2. Predict velocity (writes into u_star, v_star)
        Kokkos::fence(); auto t1 = Clock::now();
        integrator.predict(state, bc, u_star, v_star, config.re, dt);

        // 3. Compute pressure Poisson RHS from divergence of u_star
        Kokkos::fence(); auto t2 = Clock::now();
        physics::compute_pressure_rhs(state, u_star, v_star, dt, rhs);

        // 4. Solve pressure Poisson equation (reads rhs, writes s.p)
        Kokkos::fence(); auto t3 = Clock::now();
        PressureSolveResult pres = pressure.solve(state, rhs);

        // 5. Correct velocity with pressure gradient
        Kokkos::fence(); auto t4 = Clock::now();
        physics::correct_velocity(state, u_star, v_star, dt);
        Kokkos::fence(); auto t5 = Clock::now();

        stats_.wall_bc             += dur(t0, t1);
        stats_.wall_predict        += dur(t1, t2);
        stats_.wall_pressure_rhs   += dur(t2, t3);
        stats_.wall_pressure_solve += dur(t3, t4);
        stats_.wall_correct        += dur(t4, t5);

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

        ++stats_.n_steps;
        stats_.dt_sum += dt;
        stats_.dt_min  = std::min(stats_.dt_min, dt);
        stats_.dt_max  = std::max(stats_.dt_max, dt);

        stats_.pres_iters_min    = std::min(stats_.pres_iters_min, pres.iters);
        stats_.pres_iters_max    = std::max(stats_.pres_iters_max, pres.iters);
        stats_.pres_iters_total += pres.iters;
        stats_.pres_res_max      = std::max(stats_.pres_res_max, pres.final_residual);

        if constexpr (std::is_same_v<Integrator, CrankThatNicolson>) {
            stats_.has_rbgs = true;
            const int    u_it  = integrator.last_u_result.iters;
            const int    v_it  = integrator.last_v_result.iters;
            const double u_res = integrator.last_u_result.final_residual;
            const double v_res = integrator.last_v_result.final_residual;
            const int both_min = std::min(u_it, v_it);
            const int both_max = std::max(u_it, v_it);
            stats_.rbgs_iters_min    = (stats_.rbgs_iters_total == 0)
                                       ? both_min
                                       : std::min(stats_.rbgs_iters_min, both_min);
            stats_.rbgs_iters_max    = std::max(stats_.rbgs_iters_max,  both_max);
            stats_.rbgs_iters_total += u_it + v_it;
            stats_.rbgs_res_max      = std::max(stats_.rbgs_res_max, std::max(u_res, v_res));
        }

        state.time += dt;
        ++state.step;
    }

    void run() {
        CSVOutput output(config.output_path());

        auto wall_start = std::chrono::steady_clock::now();

        while (state.time < config.t_end && state.step < state.n_steps) {
            advance();
        }

        stats_.wall_total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();

        output.write(state);
        output.write_kinetic_energy(state);
        output.write_l2_divergence(state);
        if (config.diagnostics) {
            output.write_exact(state, config.re);
            output.write_error_norms(state);
        }
        if (config.profile) {
            output.write_run_stats(stats_, config);
        }

        print_report(stats_.wall_total);
    }

    void print_report(double wall_s) const {
        const int n = stats_.n_steps;
        std::cout << "\n=== Run Summary ===\n";
        std::cout << std::left << std::setw(28) << "  Grid"
                  << config.nx << " x " << config.ny << "\n";
        std::cout << std::setw(28) << "  Re"        << config.re   << "\n";
        std::cout << std::setw(28) << "  t_end"     << config.t_end << "\n";
        std::cout << std::setw(28) << "  Steps"     << n           << "\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(28) << "  Wall time (s)"  << wall_s << "\n";
        std::cout << std::setw(28) << "  Wall time/step (ms)"
                  << (wall_s / n * 1e3) << "\n";

        std::cout << "\n  Time step (CFL=" << config.cfl << "):\n";
        std::cout << std::setprecision(6);
        std::cout << std::setw(28) << "    dt min"  << stats_.dt_min << "\n";
        std::cout << std::setw(28) << "    dt mean" << stats_.dt_sum / n << "\n";
        std::cout << std::setw(28) << "    dt max"  << stats_.dt_max << "\n";

        if constexpr (std::is_same_v<Integrator, CrankThatNicolson>) {
            std::cout << "\n  RBGS viscous solver (tol=" << std::scientific
                      << integrator.viscous_solver.tol << ", max_iters="
                      << integrator.viscous_solver.max_iters << "):\n";
            std::cout << std::fixed << std::setprecision(2);
            std::cout << std::setw(28) << "    iters min"  << stats_.rbgs_iters_min << "\n";
            std::cout << std::setw(28) << "    iters mean"
                      << static_cast<double>(stats_.rbgs_iters_total) / (2 * n) << "\n";
            std::cout << std::setw(28) << "    iters max"  << stats_.rbgs_iters_max << "\n";
            std::cout << std::scientific << std::setprecision(2);
            std::cout << std::setw(28) << "    max residual" << stats_.rbgs_res_max << "\n";
        }

        std::cout << std::fixed;
        std::cout << "\n  Pressure multigrid (tol=" << std::scientific
                  << pressure.tol << ", max_vcycles=" << pressure.max_vcycles << "):\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(28) << "    vcycles min"  << stats_.pres_iters_min << "\n";
        std::cout << std::setw(28) << "    vcycles mean"
                  << static_cast<double>(stats_.pres_iters_total) / n << "\n";
        std::cout << std::setw(28) << "    vcycles max"  << stats_.pres_iters_max << "\n";
        std::cout << std::scientific << std::setprecision(2);
        std::cout << std::setw(28) << "    max residual" << stats_.pres_res_max << "\n";

        // Per-stage timing breakdown
        auto pct = [&](double t) { return 100.0 * t / wall_s; };
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n  Per-stage wall time:\n";
        std::cout << std::setw(28) << "    BC (s)"            << stats_.wall_bc             << "  (" << std::setprecision(1) << pct(stats_.wall_bc)             << "%)\n";
        std::cout << std::setprecision(3) << std::setw(28) << "    Predict (s)"       << stats_.wall_predict         << "  (" << std::setprecision(1) << pct(stats_.wall_predict)         << "%)\n";
        std::cout << std::setprecision(3) << std::setw(28) << "    Pressure RHS (s)"  << stats_.wall_pressure_rhs    << "  (" << std::setprecision(1) << pct(stats_.wall_pressure_rhs)    << "%)\n";
        std::cout << std::setprecision(3) << std::setw(28) << "    Pressure solve (s)"<< stats_.wall_pressure_solve  << "  (" << std::setprecision(1) << pct(stats_.wall_pressure_solve)  << "%)\n";
        std::cout << std::setprecision(3) << std::setw(28) << "    Correct (s)"       << stats_.wall_correct         << "  (" << std::setprecision(1) << pct(stats_.wall_correct)         << "%)\n";
        std::cout << "\n";
    }
};
