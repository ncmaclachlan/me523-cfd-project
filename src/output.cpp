#include "output.hpp"

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

void CSVOutput::write_kinetic_energy(const SimState& s, double dt) const {
    Kokkos::fence();

    auto ke_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, s.ke_history);

    std::ofstream f(this->filename + "_ke.csv");
    if (!f.is_open())
        throw std::runtime_error("Cannot open output file: " + this->filename + "_ke.csv");
    f << std::scientific << std::setprecision(10);
    f << "step,time,kinetic_energy\n";
    for (int n = 0; n < s.step; ++n) {
        f << n << "," << (n + 1) * dt << "," << ke_h(n) << "\n";
    }

    std::cout << "Kinetic energy history written to " << this->filename << "_ke.csv\n";
}

void CSVOutput::write_l2_divergence(const SimState& s, double dt) const {
    Kokkos::fence();

    auto div_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, s.div_history);

    std::ofstream f(this->filename + "_div.csv");
    if (!f.is_open())
        throw std::runtime_error("Cannot open output file: " + this->filename + "_div.csv");
    f << std::scientific << std::setprecision(10);
    f << "step,time,l2_divergence\n";
    for (int n = 0; n < s.step; ++n) {
        f << n << "," << (n + 1) * dt << "," << div_h(n) << "\n";
    }

    std::cout << "L2 divergence history written to " << this->filename << "_div.csv\n";
}
