#pragma once
#include "sim_state.hpp"
#include "run_config.hpp"
#include <string>

struct CSVOutput {
    std::string filename;

    explicit CSVOutput(std::string fname);
    void write(const SimState& s) const;
};
