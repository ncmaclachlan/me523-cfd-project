#pragma once

#include "grid.hpp"

#include <cstddef>
#include <stdexcept>

class MacGrid2D : public Grid {
private:
    int nx_ = 0;
    int ny_ = 0;
    double lx_ = 0.0;
    double ly_ = 0.0;
    double dx_ = 0.0;
    double dy_ = 0.0;

public: 

    MacGrid2D(int nx, int ny, double lx, double ly)
    : nx_(nx), ny_(ny), lx_(lx), ly_(ly)
    {
        if (nx_ <= 0) {
            throw std::invalid_argument("MacGrid2D: nx must be positive");
        }
        if (ny_ <= 0) {
            throw std::invalid_argument("MacGrid2D: ny must be positive");
        }
        if (lx_ <= 0.0) {
            throw std::invalid_argument("MacGrid2D: lx must be positive");
        }
        if (ly_ <= 0.0) {
            throw std::invalid_argument("MacGrid2D: ly must be positive");
        }
        dx_ = lx_ / static_cast<double>(nx_);
        dy_ = ly_ / static_cast<double>(ny_);
    }

    // overide base class
    std::size_t dim() const override { return 2; }

    int cells(std::size_t axis) const override {
        switch (axis) {
            case 0: return nx_;
            case 1: return ny_;
            default:
                throw std::out_of_range("MacGrid2D::cells: axis out of range");
        }
    }

    double domain_length(std::size_t axis) const override {
        switch (axis) {
            case 0: return lx_;
            case 1: return ly_;
            default:
                throw std::out_of_range("MacGrid2D::domain_length: axis out of range");
        }
    }

    double spacing(std::size_t axis) const override {
        switch (axis) {
            case 0: return dx_;
            case 1: return dy_;
            default:
                throw std::out_of_range("MacGrid2D::spacing: axis out of range");
        }
    }

    // accessors
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    double lx() const { return lx_; }
    double ly() const { return ly_; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }

    // cell centered pressures
    // pij = p(dx*i, dy*j)
    int p_nx() const { return nx_; }
    int p_ny() const { return ny_; }
    
    // face centered velocities
    // u_faceij = u(dx*(i+1/2), dy*j) 
    int u_nx() const { return nx_ + 1; }
    int u_ny() const { return ny_; }

    int v_nx() const { return nx_; }
    int v_ny() const { return ny_ + 1; }

};