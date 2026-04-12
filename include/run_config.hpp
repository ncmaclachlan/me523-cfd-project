#pragma once
#include <string>

struct RunConfig {
    int    nx = 64;
    int    ny = 64;
    double lx = 1.0;
    double ly = 1.0;

    double dt    = 1e-3;
    double t_end = 1.0;
    double re    = 100.0;

    int         output_interval = 100;
    std::string output_filename = "output.csv";
};
