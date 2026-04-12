#include "output.hpp"

#include <Kokkos_Core.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Parse a CSV file into a vector of rows, each row a vector of strings.
static std::vector<std::vector<std::string>> read_csv(const std::string& path) {
    std::ifstream f(path);
    assert(f.is_open());
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(f, line)) {
        std::vector<std::string> cols;
        std::istringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ','))
            cols.push_back(cell);
        rows.push_back(std::move(cols));
    }
    return rows;
}

static void test_row_counts() {
    // 2x3 grid → p: 2*3=6 rows, u: 3*3=9 rows, v: 2*4=8 rows (plus header)
    constexpr int nx = 2, ny = 3;
    MacGrid2D grid(nx, ny, 1.0, 1.0);
    SimState state(grid);

    CSVOutput out("test_row_counts");
    out.write(state);

    auto p_csv = read_csv("test_row_counts_p.csv");
    auto u_csv = read_csv("test_row_counts_u.csv");
    auto v_csv = read_csv("test_row_counts_v.csv");

    assert(p_csv.size() == 1 + nx * ny);           // header + 6
    assert(u_csv.size() == 1 + (nx + 1) * ny);     // header + 9
    assert(v_csv.size() == 1 + nx * (ny + 1));      // header + 8

    std::remove("test_row_counts_p.csv");
    std::remove("test_row_counts_u.csv");
    std::remove("test_row_counts_v.csv");

    std::cout << "PASS: test_row_counts\n";
}

static void test_pressure_values() {
    constexpr int nx = 2, ny = 2;
    constexpr double lx = 1.0, ly = 1.0;
    MacGrid2D grid(nx, ny, lx, ly);
    SimState state(grid);

    // Fill pressure on host, then deep_copy to device view
    auto p_h = Kokkos::create_mirror_view(state.p);
    for (int i = grid.p_i_begin(); i < grid.p_i_end(); ++i)
        for (int j = grid.p_j_begin(); j < grid.p_j_end(); ++j)
            p_h(i, j) = 100.0 * (i - grid.ng) + (j - grid.ng);  // deterministic pattern
    Kokkos::deep_copy(state.p, p_h);

    CSVOutput out("test_pval");
    out.write(state);

    auto csv = read_csv("test_pval_p.csv");
    assert(csv.size() == 1 + nx * ny);
    assert(csv[0][4] == "p");  // header check

    const double dx = lx / nx;
    const double dy = ly / ny;

    // Verify every data row
    for (size_t r = 1; r < csv.size(); ++r) {
        int i  = std::stoi(csv[r][0]);
        int j  = std::stoi(csv[r][1]);
        double x  = std::stod(csv[r][2]);
        double y  = std::stod(csv[r][3]);
        double pv = std::stod(csv[r][4]);

        double expect_x = (i + 0.5) * dx;
        double expect_y = (j + 0.5) * dy;
        double expect_p = 100.0 * i + j;

        assert(std::fabs(x  - expect_x) < 1e-8);
        assert(std::fabs(y  - expect_y) < 1e-8);
        assert(std::fabs(pv - expect_p) < 1e-8);
    }

    std::remove("test_pval_p.csv");
    std::remove("test_pval_u.csv");
    std::remove("test_pval_v.csv");

    std::cout << "PASS: test_pressure_values\n";
}

static void test_velocity_coordinates() {
    constexpr int nx = 3, ny = 2;
    constexpr double lx = 3.0, ly = 2.0;
    MacGrid2D grid(nx, ny, lx, ly);
    SimState state(grid);

    CSVOutput out("test_vel_coord");
    out.write(state);

    const double dx = lx / nx;
    const double dy = ly / ny;

    // u lives on x-faces: x = i*dx, y = (j+0.5)*dy
    {
        auto csv = read_csv("test_vel_coord_u.csv");
        assert(csv[0][4] == "u");
        for (size_t r = 1; r < csv.size(); ++r) {
            int i = std::stoi(csv[r][0]);
            int j = std::stoi(csv[r][1]);
            double x = std::stod(csv[r][2]);
            double y = std::stod(csv[r][3]);
            assert(std::fabs(x - i * dx) < 1e-8);
            assert(std::fabs(y - (j + 0.5) * dy) < 1e-8);
        }
    }

    // v lives on y-faces: x = (i+0.5)*dx, y = j*dy
    {
        auto csv = read_csv("test_vel_coord_v.csv");
        assert(csv[0][4] == "v");
        for (size_t r = 1; r < csv.size(); ++r) {
            int i = std::stoi(csv[r][0]);
            int j = std::stoi(csv[r][1]);
            double x = std::stod(csv[r][2]);
            double y = std::stod(csv[r][3]);
            assert(std::fabs(x - (i + 0.5) * dx) < 1e-8);
            assert(std::fabs(y - j * dy) < 1e-8);
        }
    }

    std::remove("test_vel_coord_p.csv");
    std::remove("test_vel_coord_u.csv");
    std::remove("test_vel_coord_v.csv");

    std::cout << "PASS: test_velocity_coordinates\n";
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        test_row_counts();
        test_pressure_values();
        test_velocity_coordinates();
    }
    Kokkos::finalize();
    std::cout << "All output tests passed.\n";
    return 0;
}
