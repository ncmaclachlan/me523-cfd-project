#include "output.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

CSVOutput::CSVOutput(std::string fname) : filename(std::move(fname)) {}

void CSVOutput::write(const SimState& s) const {
    // TODO: write u, v, p fields to CSV
    std::ofstream f(this->filename);

    if (!f.is_open())
        throw std::runtime_error("Cannot open output file: " + this->filename);

    f << std::scientific << std::setprecision(10);
    f << "x,y,z\n"; 
    std::cout << "Output written to " << this->filename << "\n";

    f.close();
}
