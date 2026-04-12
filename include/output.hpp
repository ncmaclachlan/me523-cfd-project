#pragma once
#include "sim_state.hpp"
#include <string>

struct CSVOutput {
    std::string filename;

    explicit CSVOutput(std::string fname) : filename(std::move(fname)) {}

    template<typename P>
    void write(const SimState<P>& s) const {
        // TODO: write u, v, p fields to CSV
    }
};
