#pragma once
#include "sim_state.hpp"
#include "run_config.hpp"
#include "run_stats.hpp"
#include <string>

struct CSVOutput {
    std::string filename;

    explicit CSVOutput(std::string fname);
    void write(const SimState& s) const;
    void write_vorticity(const SimState& s) const;
    void write_exact(const SimState& s, double re) const;
    void write_kinetic_energy(const SimState& s) const;
    void write_l2_divergence(const SimState& s) const;
    void write_error_norms(const SimState& s) const;
    void write_run_stats(const RunStats& stats, const RunConfig& cfg) const;
};
