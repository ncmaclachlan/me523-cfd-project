#include "output.hpp"

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>

CSVOutput::CSVOutput(std::string fname) : filename(std::move(fname)) {}

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
