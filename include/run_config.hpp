#pragma once
#include <string>
#include <sstream>
#include <cmath>

struct RunConfig {
    int    nx = 64;
    int    ny = 64;
    double lx = 2.0 * M_PI;
    double ly = 2.0 * M_PI;

    double dt    = 1e-3;   // used only when cfl <= 0
    double cfl   = 0.5;   // adaptive CFL number; set <= 0 to use fixed dt
    double t_end = 10.0;
    double re    = 100.0;

    int         output_interval = 100;
    std::string output_filename = "output";

    /// Build the run directory path: data/run_{nx}_{ny}_{Re}_{dt}
    std::string run_dir() const {
        std::ostringstream oss;
        oss << "data/run_" << nx << "_" << ny << "_" << re << "_" << dt;
        return oss.str();
    }

    /// Full path for output files: {run_dir}/output/{output_filename}
    std::string output_path() const {
        return run_dir() + "/output/" + output_filename;
    }
};
