#include "output.hpp"

CSVOutput::CSVOutput(std::string fname) : filename(std::move(fname)) {}

void CSVOutput::write(const SimState& s) const {
    // TODO: write u, v, p fields to CSV
}
