#include "output.hpp"
#include "run_stats.hpp"

#include <Kokkos_Core.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

CSVOutput::CSVOutput(std::string fname) : filename(std::move(fname)) {
    auto parent = std::filesystem::path(filename).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void CSVOutput::write(const SimState& s) const {
    Kokkos::fence();

    auto u_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, s.u);
    auto v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, s.v);
    auto p_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, s.p);

    const auto& g = s.grid;

    // --- pressure (cell-centred) ---
    {
        std::ofstream f(this->filename + "_p.csv");
        if (!f.is_open())
            throw std::runtime_error("Cannot open output file: " + this->filename + "_p.csv");
        f << std::scientific << std::setprecision(10);
        f << "i,j,x,y,p\n";
        for (int i = g.p_i_begin(); i < g.p_i_end(); ++i)
            for (int j = g.p_j_begin(); j < g.p_j_end(); ++j) {
                double x = (i - g.ng + 0.5) * g.dx;
                double y = (j - g.ng + 0.5) * g.dy;
                f << (i - g.ng) << "," << (j - g.ng) << ","
                  << x << "," << y << "," << p_h(i, j) << "\n";
            }
    }

    // --- u velocity (x-face-centred) ---
    {
        std::ofstream f(this->filename + "_u.csv");
        if (!f.is_open())
            throw std::runtime_error("Cannot open output file: " + this->filename + "_u.csv");
        f << std::scientific << std::setprecision(10);
        f << "i,j,x,y,u\n";
        for (int i = g.u_i_begin(); i < g.u_i_end(); ++i)
            for (int j = g.u_j_begin(); j < g.u_j_end(); ++j) {
                double x = (i - g.ng) * g.dx;
                double y = (j - g.ng + 0.5) * g.dy;
                f << (i - g.ng) << "," << (j - g.ng) << ","
                  << x << "," << y << "," << u_h(i, j) << "\n";
            }
    }

    // --- v velocity (y-face-centred) ---
    {
        std::ofstream f(this->filename + "_v.csv");
        if (!f.is_open())
            throw std::runtime_error("Cannot open output file: " + this->filename + "_v.csv");
        f << std::scientific << std::setprecision(10);
        f << "i,j,x,y,v\n";
        for (int i = g.v_i_begin(); i < g.v_i_end(); ++i)
            for (int j = g.v_j_begin(); j < g.v_j_end(); ++j) {
                double x = (i - g.ng + 0.5) * g.dx;
                double y = (j - g.ng) * g.dy;
                f << (i - g.ng) << "," << (j - g.ng) << ","
                  << x << "," << y << "," << v_h(i, j) << "\n";
            }
    }

    std::cout << "Output written to " << this->filename << "_{p,u,v}.csv\n";
}

void CSVOutput::write_exact(const SimState& s, double re) const {
    const auto& g = s.grid;
    const double nu = 1.0 / re;
    const double decay = std::exp(-2.0 * nu * s.time);

    // --- exact u velocity (x-face-centred) ---
    {
        std::ofstream f(this->filename + "_u_exact.csv");
        if (!f.is_open())
            throw std::runtime_error("Cannot open output file: " + this->filename + "_u_exact.csv");
        f << std::scientific << std::setprecision(10);
        f << "i,j,x,y,u_exact\n";
        for (int i = g.u_i_begin(); i < g.u_i_end(); ++i)
            for (int j = g.u_j_begin(); j < g.u_j_end(); ++j) {
                double x = (i - g.ng) * g.dx;
                double y = (j - g.ng + 0.5) * g.dy;
                double u_exact = std::sin(x) * std::cos(y) * decay;
                f << (i - g.ng) << "," << (j - g.ng) << ","
                  << x << "," << y << "," << u_exact << "\n";
            }
    }

    // --- exact v velocity (y-face-centred) ---
    {
        std::ofstream f(this->filename + "_v_exact.csv");
        if (!f.is_open())
            throw std::runtime_error("Cannot open output file: " + this->filename + "_v_exact.csv");
        f << std::scientific << std::setprecision(10);
        f << "i,j,x,y,v_exact\n";
        for (int i = g.v_i_begin(); i < g.v_i_end(); ++i)
            for (int j = g.v_j_begin(); j < g.v_j_end(); ++j) {
                double x = (i - g.ng + 0.5) * g.dx;
                double y = (j - g.ng) * g.dy;
                double v_exact = -std::cos(x) * std::sin(y) * decay;
                f << (i - g.ng) << "," << (j - g.ng) << ","
                  << x << "," << y << "," << v_exact << "\n";
            }
    }

    // --- exact pressure (cell-centred) ---
    {
        std::ofstream f(this->filename + "_p_exact.csv");
        if (!f.is_open())
            throw std::runtime_error("Cannot open output file: " + this->filename + "_p_exact.csv");
        f << std::scientific << std::setprecision(10);
        f << "i,j,x,y,p_exact\n";
        for (int i = g.p_i_begin(); i < g.p_i_end(); ++i)
            for (int j = g.p_j_begin(); j < g.p_j_end(); ++j) {
                double x = (i - g.ng + 0.5) * g.dx;
                double y = (j - g.ng + 0.5) * g.dy;
                double p_exact = -0.25 * (std::cos(2.0*x) + std::cos(2.0*y)) * decay * decay;
                f << (i - g.ng) << "," << (j - g.ng) << ","
                  << x << "," << y << "," << p_exact << "\n";
            }
    }

    std::cout << "Exact solution written to " << this->filename << "_{u,v,p}_exact.csv\n";
}

void CSVOutput::write_kinetic_energy(const SimState& s) const {
    std::ofstream f(this->filename + "_ke.csv");
    if (!f.is_open())
        throw std::runtime_error("Cannot open output file: " + this->filename + "_ke.csv");
    f << std::scientific << std::setprecision(10);
    f << "step,time,kinetic_energy\n";
    for (int n = 0; n < s.step; ++n) {
        f << n << "," << s.time_history(n) << "," << s.ke_history(n) << "\n";
    }

    std::cout << "Kinetic energy history written to " << this->filename << "_ke.csv\n";
}

void CSVOutput::write_l2_divergence(const SimState& s) const {
    std::ofstream f(this->filename + "_div.csv");
    if (!f.is_open())
        throw std::runtime_error("Cannot open output file: " + this->filename + "_div.csv");
    f << std::scientific << std::setprecision(10);
    f << "step,time,l2_divergence\n";
    for (int n = 0; n < s.step; ++n) {
        f << n << "," << s.time_history(n) << "," << s.div_history(n) << "\n";
    }

    std::cout << "L2 divergence history written to " << this->filename << "_div.csv\n";
}

void CSVOutput::write_error_norms(const SimState& s) const {
    std::ofstream f(this->filename + "_error.csv");
    if (!f.is_open())
        throw std::runtime_error("Cannot open output file: " + this->filename + "_error.csv");
    f << std::scientific << std::setprecision(10);
    f << "step,time,l2_error,linf_error\n";
    for (int n = 0; n < s.step; ++n) {
        f << n << "," << s.time_history(n) << ","
          << s.err_l2_history(n) << "," << s.err_linf_history(n) << "\n";
    }

    std::cout << "Error norm history written to " << this->filename << "_error.csv\n";
}

void CSVOutput::write_run_stats(const RunStats& stats, const RunConfig& cfg) const {
    namespace fs = std::filesystem;
    // filename is run_dir/output/output — go up two levels to reach run_dir
    fs::path run_dir = fs::path(filename).parent_path().parent_path();
    fs::create_directories(run_dir);

    fs::path json_path = run_dir / "run_stats.json";
    std::ofstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + json_path.string());

    const std::string backend = Kokkos::DefaultExecutionSpace::name();
    const int n = stats.n_steps;
    const double wall = stats.wall_total;

    f << std::fixed << std::setprecision(6);
    f << "{\n";

    f << "  \"config\": {\n";
    f << "    \"nx\": "       << cfg.nx    << ",\n";
    f << "    \"ny\": "       << cfg.ny    << ",\n";
    f << "    \"re\": "       << cfg.re    << ",\n";
    f << "    \"cfl\": "      << cfg.cfl   << ",\n";
    f << "    \"dt\": "       << cfg.dt    << ",\n";
    f << "    \"t_end\": "    << cfg.t_end << ",\n";
    f << "    \"backend\": \"" << backend  << "\"\n";
    f << "  },\n";

    f << "  \"timing\": {\n";
    f << "    \"wall_total_s\": "          << wall                               << ",\n";
    f << "    \"wall_per_step_ms\": "      << (n > 0 ? wall / n * 1e3 : 0.0)    << ",\n";
    f << "    \"wall_bc_s\": "             << stats.wall_bc                      << ",\n";
    f << "    \"wall_predict_s\": "        << stats.wall_predict                 << ",\n";
    f << "    \"wall_pressure_rhs_s\": "   << stats.wall_pressure_rhs            << ",\n";
    f << "    \"wall_pressure_solve_s\": " << stats.wall_pressure_solve          << ",\n";
    f << "    \"wall_correct_s\": "        << stats.wall_correct                 << "\n";
    f << "  },\n";

    f << "  \"solver_stats\": {\n";
    f << "    \"n_steps\": "          << n                                               << ",\n";
    f << "    \"dt_min\": "           << stats.dt_min                                    << ",\n";
    f << "    \"dt_mean\": "          << (n > 0 ? stats.dt_sum / n : 0.0)               << ",\n";
    f << "    \"dt_max\": "           << stats.dt_max                                    << ",\n";
    f << "    \"pres_iters_min\": "   << (stats.pres_iters_min == std::numeric_limits<int>::max() ? 0 : stats.pres_iters_min) << ",\n";
    f << "    \"pres_iters_mean\": "  << (n > 0 ? static_cast<double>(stats.pres_iters_total) / n : 0.0) << ",\n";
    f << "    \"pres_iters_max\": "   << stats.pres_iters_max                           << ",\n";
    f << "    \"pres_res_max\": "     << stats.pres_res_max                             << ",\n";
    f << "    \"has_rbgs\": "         << (stats.has_rbgs ? "true" : "false")            << ",\n";
    f << "    \"rbgs_iters_min\": "   << stats.rbgs_iters_min                           << ",\n";
    f << "    \"rbgs_iters_max\": "   << stats.rbgs_iters_max                           << ",\n";
    f << "    \"rbgs_iters_total\": " << stats.rbgs_iters_total                         << ",\n";
    f << "    \"rbgs_res_max\": "     << stats.rbgs_res_max                             << "\n";
    f << "  }\n";
    f << "}\n";

    std::cout << "Run stats written to " << json_path.string() << "\n";
}
